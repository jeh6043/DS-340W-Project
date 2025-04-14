import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import cosine_similarity
import gc


def approx_inverse_hessian(model, dataloader, loss_fn, damping=1e-2):
    model.eval()
    grad_sq_sum = None
    count = 0
    for batch in dataloader:
        print(count + 1)
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(model.device)

        model.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        grad = []
        for param in model.parameters():
            if param.grad is not None:
                grad.append(param.grad.detach().cpu().flatten())

        grad_vector = torch.cat(grad)
        grad_sq = grad_vector ** 2

        if grad_sq_sum is None:
            grad_sq_sum = grad_sq
        else:
            grad_sq_sum += grad_sq

        count += 1
        torch.cuda.empty_cache()

    grad_norms = grad_sq_sum / count
    H_inv_diag = 1.0 / (grad_norms + damping)
    H_inv_diag = H_inv_diag / torch.norm(H_inv_diag)


    return H_inv_diag

def compute_influence_scores(model, dataloader, loss_fn, H_inv, batch_size=1):
    influences = []
    
    # Process each batch of gradients
    for i, batch in enumerate(dataloader):
        # Move the batch to the appropriate device
        inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(model.device)

        model.zero_grad()
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()

        grad = []
        for param in model.parameters():
            if param.grad is not None:
                grad.append(param.grad.detach().cpu().flatten())
        
        grad_vector = torch.cat(grad)

        # Calculate influence for the current batch
        influence = H_inv * grad_vector

        influences.append(influence)
        
        # Optional: periodically clear memory
        torch.cuda.empty_cache()

    return influences


def batched_influence_pruning(model, full_dataset, loss_fn, H_inv, batch_size=100, epsilon=0.95):
    selected_indices = []
    cumulative_influence = None
    count = 0

    for i in range(0, len(full_dataset), batch_size):
        print(f"[INFO] Processing batch {i} to {min(i + batch_size, len(full_dataset))}")
        batch_indices = list(range(i, min(i + batch_size, len(full_dataset))))
        batch_loader = DataLoader(Subset(full_dataset, batch_indices), batch_size=1)

        influences = compute_influence_scores(model, batch_loader, loss_fn, H_inv)

        for j, infl in zip(batch_indices, influences):
            infl = infl / (torch.norm(infl) + 1e-8)  # normalize influence

            if cumulative_influence is None:
                cumulative_influence = infl.clone()
                selected_indices.append(j)
                continue

            similarity = cosine_similarity(cumulative_influence.unsqueeze(0), infl.unsqueeze(0)).item()
            print(f"Cosine similarity with cumulative influence: {similarity:.4f}")

            if similarity < epsilon:
                selected_indices.append(j)
                cumulative_influence += infl  # accumulate influence vector

            # Cleanup
            del infl
            gc.collect()
            torch.cuda.empty_cache()

        del batch_loader, influences
        gc.collect()
        torch.cuda.empty_cache()

    return selected_indices