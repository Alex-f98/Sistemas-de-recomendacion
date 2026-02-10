import torch
import numpy as np

from utils import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    apk,
    ndcg_at_k,
    mae,
    mse,
    rmse
)

class evaluatorMF:

    def __init__(self, model, device="cpu"):
        self.model  = model
        self.device = device

    def evaluate_ratings(self, data_loader, p=0.5):
        self.model.eval()

        preds = []
        trues = []

        with torch.no_grad():
            for items, users, ratings in data_loader:
                users   = users.to(self.device)
                items   = items.to(self.device)
                ratings = ratings.to(self.device).float()

                pred = self.model.predict(users, items, p)
                preds.append(pred)
                trues.append(ratings)

        preds = torch.cat(preds)
        trues = torch.cat(trues)

        return {
            "MSE": mse(preds, trues).item(),
            "RMSE": rmse(preds, trues).item(),
            "MAE": mae(preds, trues).item(),
        }

    # ============================
    # TOP-K RANKING METRICS
    # ============================
    def recommend_for_user(self, user, all_items, p=0.5):
        """Devuelve todos los items ordenados por puntaje para un usuario."""
        user_tensor  = torch.tensor([user],    dtype=torch.long).to(self.device)
        item_tensor  = torch.tensor(all_items, dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model.predict(
                user_tensor.repeat(len(item_tensor)),
                item_tensor,
                p
            )
        scores = scores.cpu().numpy()
        ranking = np.argsort(-scores)  # Descendente
        return ranking

    def evaluate_ranking(self, test_interactions, all_items, k=10):
        """
        test_interactions: dict {user_id: [relevant_items]}
        """
        precisions, recalls, hits, ndcgs, maps = [], [], [], [], []

        for user, relevant_items in test_interactions.items():
            recs = self.recommend_for_user(user, all_items)

            precisions.append(precision_at_k(recs, relevant_items, k))
            recalls.append(recall_at_k(recs, relevant_items, k))
            hits.append(hit_rate_at_k(recs, relevant_items, k))
            ndcgs.append(ndcg_at_k(recs, relevant_items, k))
            maps.append(apk(recs, relevant_items, k))

        return {
            f"Precision@{k}" : float(np.mean(precisions)),
            f"Recall@{k}"    : float(np.mean(recalls)),
            f"HitRate@{k}"   : float(np.mean(hits)),
            f"NDCG@{k}"      : float(np.mean(ndcgs)),
            f"MAP@{k}"       : float(np.mean(maps)),
        }
