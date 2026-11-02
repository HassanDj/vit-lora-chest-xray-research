"""Training pipeline: LP vs LoRA, PneumoniaMNIST+ (224px), ViT-B/16.

Run in Colab GPU; does not evaluate the test split until *all four*
validation-selected checkpoints exist. No experimental result is fabricated.
"""
from __future__ import annotations
import gc
import json
import random
import time
from pathlib import Path

import medmnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from medmnist import PneumoniaMNIST
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoImageProcessor, ViTModel

from utils import (SEED, nested_stratified_indices, class_weights,
                   binary_metrics, should_extend_after_three,
                   should_stop_extension)

CHECKPOINT = 'google/vit-base-patch16-224-in21k'
BATCH_SIZE = 16
RESULTS = Path('results')
CHECKPOINTS = RESULTS / '.checkpoints'
EXPERIMENTS = [('linear_probing', 400), ('lora', 400),
               ('linear_probing', 1200), ('lora', 1200)]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic selection/init; GPU operations can still vary by platform.


class WrappedMedMNIST(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, int(np.asarray(label).reshape(-1)[0])


def prepare_data():
    """Load official MedMNIST+ 224px splits, never mix their numeric indices."""
    set_seed()
    processor = AutoImageProcessor.from_pretrained(CHECKPOINT, use_fast=False)

    def transform(pil_image):
        image = pil_image.convert('RGB')
        return processor(images=image, return_tensors='pt')['pixel_values'][0]

    datasets = {}
    for split in ('train', 'val', 'test'):
        raw = PneumoniaMNIST(split=split, size=224, download=True, transform=transform)
        datasets[split] = WrappedMedMNIST(raw)
    assert [len(datasets[x]) for x in ('train', 'val', 'test')] == [4708, 524, 624]

    y_train = np.asarray(datasets['train'].dataset.labels).reshape(-1).astype(int)
    indices = nested_stratified_indices(y_train)
    for size in (400, 1200):
        print(f'Train subset n={size}, class counts NORMAL/PNEUMONIA:',
              np.bincount(y_train[indices[size]], minlength=2).tolist())
    image, label = datasets['train'][0]
    assert tuple(image.shape) == (3, 224, 224), tuple(image.shape)
    assert label in (0, 1)
    print('Splits: train=4708, validation=524, test=624')
    print('Image tensor:', tuple(image.shape), '| first class:', label)
    print('400 indices nested within 1200:', set(indices[400]).issubset(set(indices[1200])))
    print('Patient-level independence is NOT verified by these checks.')

    return {'datasets': datasets, 'indices': indices, 'train_labels': y_train,
            'processor': processor}


class ViTClassifier(nn.Module):
    """Same external 768->2 head in LP and LoRA, independent of PEFT."""
    def __init__(self, method):
        super().__init__()
        if method not in ('linear_probing', 'lora'):
            raise ValueError(method)
        self.method = method
        self.encoder = ViTModel.from_pretrained(CHECKPOINT, add_pooling_layer=False)
        # Reset *before* initializing the external head, guaranteeing same init.
        torch.manual_seed(SEED)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)
        if method == 'linear_probing':
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
        else:
            self.encoder = get_peft_model(
                self.encoder,
                LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                           target_modules=['query', 'value'], bias='none'))
        self.check_trainability()

    def check_trainability(self):
        assert all(p.requires_grad for p in self.classifier.parameters())
        if self.method == 'linear_probing':
            assert not any(p.requires_grad for p in self.encoder.parameters())
        else:
            enc_params = list(self.encoder.named_parameters())
            assert any('lora_' in name and p.requires_grad for name, p in enc_params)
            assert all(not p.requires_grad for name, p in enc_params if 'lora_' not in name)

    def forward(self, pixel_values):
        cls_token = self.encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


def make_loader(ds, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for images, y in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        assert logits.ndim == 2 and logits.shape[1] == 2
        positive = torch.softmax(logits.float(), dim=-1)[:, 1]
        assert bool(((positive >= 0) & (positive <= 1)).all())
        labels.extend(y.numpy().tolist())
        probabilities.extend(positive.detach().cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def train_one(method, budget, data):
    """Train/checkpoint using train and validation only (never test)."""
    if not torch.cuda.is_available():
        raise RuntimeError('Please choose a GPU runtime (Colab: Runtime > Change runtime type > T4 GPU).')
    if (method, budget) not in EXPERIMENTS:
        raise ValueError((method, budget))
    set_seed()
    RESULTS.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    run_id = f'{method}_{budget}'
    ckpt_file = CHECKPOINTS / f'{run_id}.pt'
    log_file = RESULTS / 'validation_history.csv'
    runs_file = RESULTS / 'run_summaries.csv'
    device = torch.device('cuda')

    idx = data['indices'][budget]
    y_budget = data['train_labels'][idx]
    weights = torch.as_tensor(class_weights(y_budget), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = make_loader(Subset(data['datasets']['train'], idx.tolist()), shuffle=True)
    val_loader = make_loader(data['datasets']['val'])

    model = ViTClassifier(method).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=0.001 if method == 'linear_probing' else 0.0002,
                            weight_decay=0.01)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nStarting {run_id}: {trainable_params:,}/{total_params:,} trainable parameters')
    history = []
    best_auc = float('-inf')
    best_epoch = None
    optimizer_updates = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    epoch = 1
    extend = False

    while epoch <= 6:
        model.train()
        if method == 'linear_probing':
            # model.train() re-enables dropout; frozen backbone must be eval().
            model.encoder.eval()
            assert not model.encoder.training
        running_loss = 0.0
        n = 0
        for images, target in train_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, target)
            loss.backward()
            opt.step()
            optimizer_updates += 1
            running_loss += loss.item() * len(target)
            n += len(target)
        val_labels, val_scores = predict(model, val_loader, device)
        val_auc = binary_metrics(val_labels, val_scores)['auroc']
        history.append(float(val_auc))
        print(f'{run_id} epoch {epoch} | train_loss={running_loss/n:.4f} | val_AUROC={val_auc:.4f}')
        log_row = {'run_id':run_id, 'method':method, 'budget':budget,
                   'epoch':epoch, 'training_loss':running_loss/n, 'validation_auroc':val_auc}
        pd.DataFrame([log_row]).to_csv(log_file, index=False,
                                        mode='a', header=not log_file.exists())
        if val_auc > best_auc:
            best_auc, best_epoch = float(val_auc), epoch
            # Full state includes encoder; checkpoint saved locally but excluded from Git.
            torch.save(model.state_dict(), ckpt_file)
        if epoch == 3:
            extend = should_extend_after_three(history)
            print('Pre-registered extension after epoch 3:', extend)
            if not extend:
                break
        elif epoch >= 5 and should_stop_extension(history):
            print('Stopping: two consecutive epochs with insufficient val-AUROC gain')
            break
        epoch += 1

    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    row = {'run_id':run_id, 'method':method, 'budget':budget,
           'best_validation_auroc':best_auc, 'best_epoch':best_epoch,
           'epochs_run':len(history), 'optimizer_updates':optimizer_updates,
           'training_seconds':duration, 'peak_gpu_mib':peak_mib,
           'total_parameters':total_params, 'trainable_parameters':trainable_params,
           'trainable_pct':100 * trainable_params/total_params}
    if runs_file.exists():
        existing = pd.read_csv(runs_file)
        existing = existing[existing.run_id != run_id]
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(runs_file,index=False)
    else:
        pd.DataFrame([row]).to_csv(runs_file, index=False)
    print(f'Finished {run_id}; best val AUROC={best_auc:.4f} @ epoch {best_epoch}.')
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_experiments(data, experiments):
    for method, budget in experiments:
        train_one(method, budget, data)


def finalize_results(data):
    """Only here is the held-out test set evaluated once, after all four runs."""
    runs_file = RESULTS / 'run_summaries.csv'
    if not runs_file.exists():
        raise RuntimeError('No training runs completed')
    summary = pd.read_csv(runs_file)
    expected = {f'{method}_{budget}' for method, budget in EXPERIMENTS}
    if set(summary['run_id']) != expected or not all((CHECKPOINTS / f'{r}.pt').exists() for r in expected):
        raise RuntimeError('Complete all four train+validation experiments before consulting test')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU runtime required for evaluation')

    device = torch.device('cuda')
    loader = make_loader(data['datasets']['test'])
    results, roc_data = [], {}
    for method, budget in EXPERIMENTS:
        run_id = f'{method}_{budget}'
        model = ViTClassifier(method).to(device)
        model.load_state_dict(torch.load(CHECKPOINTS / f'{run_id}.pt', map_location=device,
                                         weights_only=True))
        y_test, score = predict(model, loader, device)
        metrics = binary_metrics(y_test, score, threshold=0.5)
        results.append({'run_id':run_id, 'method':method, 'budget':budget, **metrics})
        fpr, tpr, _ = roc_curve(y_test, score)
        roc_data[run_id] = (fpr, tpr)
        print(run_id, 'test AUROC:', f'{metrics["auroc"]:.4f}')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    out = pd.DataFrame(results).merge(summary, on=['run_id', 'method', 'budget'])
    out.to_csv(RESULTS / 'metrics.csv', index=False)
    all_aucs = out.pivot(index='budget', columns='method', values='auroc').sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in all_aucs.columns:
        ax.plot(all_aucs.index, all_aucs[col], marker='o', label=col)
    ax.set(xlabel='Labeled training images', ylabel='Test AUROC', title='Data efficiency')
    ax.legend(); fig.tight_layout(); fig.savefig(RESULTS/'auroc_vs_samples.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for key, (fpr, tpr) in roc_data.items():
        auc = float(out.loc[out.run_id==key, 'auroc'].iloc[0])
        ax.plot(fpr, tpr, label=f'{key} (AUROC {auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC curves')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(RESULTS/'roc_curves.png',dpi=160);plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(out))
    axes[0].bar(x, out['trainable_parameters'])
    axes[0].set(ylabel='Trainable parameters', title='Trainable parameters')
    axes[1].bar(x, out['training_seconds']/60)
    axes[1].set(ylabel='Minutes', title='Training runtime')
    for ax in axes:
        ax.set_xticks(x, out['run_id'], rotation=30, ha='right')
    fig.tight_layout();fig.savefig(RESULTS/'training_efficiency.png',dpi=160);plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, (_, row) in zip(axes.ravel(), out.iterrows()):
        cm = np.array([[row.tn,row.fp],[row.fn,row.tp]],dtype=int)
        ax.imshow(cm, cmap='Blues');ax.set_title(row.run_id)
        ax.set_xticks([0,1], ['Normal','Pneumonia']);ax.set_yticks([0,1], ['Normal','Pneumonia'])
        ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(RESULTS/'confusion_matrices.png',dpi=160);plt.close(fig)

    import transformers, peft, sklearn
    cfg = {'checkpoint':CHECKPOINT, 'seed':SEED, 'batch_size':BATCH_SIZE,
           'precision':'fp32', 'optimizer':'AdamW', 'weight_decay':0.01,
           'linear_probing_learning_rate':0.001, 'lora_learning_rate':0.0002,
           'lora':{'r':8,'alpha':16,'dropout':0.1,'target_modules':['query','value']},
           'selection':'maximum validation AUROC', 'test_evaluation':'one final pass',
           'train_subset_400':data['indices'][400].tolist(),
           'train_subset_1200':data['indices'][1200].tolist(),
           'library_versions':{'torch':torch.__version__, 'transformers':transformers.__version__,
                               'peft':peft.__version__, 'medmnist':medmnist.__version__,
                               'sklearn':sklearn.__version__},
           'limitations':['one seed','two budgets','one dataset','no patient split verification',
                          'different learning rates by method','no external validation']}
    with open(RESULTS/'run_config.json','w') as file:
        json.dump(cfg,file,indent=2)
    return out
