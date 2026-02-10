import torch
import numpy as np



#https://arxiv.org/html/2312.16015v2

#MSE
def mse(pred, true):
    return torch.mean((pred - true)**2)

#RMSE
def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true)**2))

#MAE
def mae(pred, true):
    return torch.mean(torch.abs(pred - true))

#Precision@k
def precision_at_k(recommended_items, relevant_items, k):
    """
    recommended_items: lista de items recomendados ordenados por relevancia
    relevant_items: lista de items relevantes
    k: número de items a considerar

    Fracción de los top-K recomendados que son relevantes
    Fórmula: hits / k donde hits = |recomendados[:k] ∩ relevantes|
    Uso: Penaliza recomendaciones irrelevantes en top-K
    """
    recommended_k = recommended_items[:k]
    hits = len(set(relevant_items).intersection(recommended_k))
    return hits / k

# def precision_at_k(predictions, k=10, threshold=4):  # Rating>4=relevant
#     tp, fp = 0, 0
#     for pred in predictions:
#         if len(pred.recs[:k]) > 0:  # Top-k
#             tp += sum(1 for rec in pred.recs[:k] if rec.rating > threshold)
#             fp += k - tp
#     return tp / (tp + fp)


#Recall@k
def recall_at_k(recommended_items, relevant_items, k):
    """
    Qué mide: Fracción de todos los relevantes que aparecen en top-K.
    Fórmula: hits / len(relevantes)
    Uso: Penaliza si te perdés relevantes (deja afuera del top-K).
    """
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(relevant_items))
    return hits / len(relevant_items) if len(relevant_items) > 0 else 0.0

#HitRate@k
def hit_rate_at_k(recommended_items, relevant_items, k):
    """
    Qué mide: Si al menos uno relevante está en top-K (binario).
    Fórmula: 1.0 si intersection > 0, else 0.0
    Uso: Métrica simple de "algo bueno en top-K".
    """
    recommended_k = recommended_items[:k]
    return 1.0 if len(set(recommended_k) & set(relevant_items)) > 0 else 0.0

#APK
def apk(recommended_items, relevant_items, k):
    """
    Qué mide: Precisión promedio en posiciones donde hay hit, ponderada por posición.
    Fórmula: Σ(hits_pos/i) / len(relevantes)
    Uso: Recompensa hits tempranos (posición importa).
    """
    recommended_k = recommended_items[:k]
    score = 0.0
    hits = 0

    for i, item in enumerate(recommended_k, start=1):
        if item in relevant_items:
            hits += 1
            score += hits / i

    return score / len(relevant_items) if relevant_items else 0.0

#MAP@K (Mean Average Precision)
def mapk(all_recommended, all_relevant, k):
    """
    Qué mide: Promedio de APK sobre todos los usuarios.
    Fórmula: MAP@5 = mean(AP_user1, AP_user2, ..., AP_userN)
    Uso: Métrica global agregada
    """
    return np.mean([
        apk(rec, rel, k)
        for rec, rel in zip(all_recommended, all_relevant)
    ])

#DCG@K (Discounted Cumulative Gain)
def dcg_at_k(recommended_items, relevant_items, k):
    """
    Qué mide: Ganancia acumulada con descuento por posición.
    Fórmula: Σ(1/log2(pos+1) si relevante)
    Nota: Asume gain=1 (binario), común en recsys.
    """
    recommended_k = recommended_items[:k]
    dcg = 0.0
    for idx, item in enumerate(recommended_k, start=1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(idx + 1)
    return dcg

#NDCG@K (Normalized Discounted Cumulative Gain)
def ndcg_at_k(recommended_items, relevant_items, k):
    """
    Qué mide: DCG dividido por el DCG ideal (todos relevantes primero).
    Fórmula: dcg_at_k(rec, rel, k) / dcg_at_k(rel, rel, k)
    Uso: Normaliza entre 0-1 independientemente de #relevantes.
    """
    ideal_dcg = dcg_at_k(relevant_items, relevant_items, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(recommended_items, relevant_items, k) / ideal_dcg


def build_test_interactions(Y, threshold=5):
    test = {}
    _, num_users = Y.shape

    for u in range(num_users):
        # indices de ítems donde el usuario u tiene rating >= threshold
        items = np.where(Y[:, u] >= threshold)[0]
        if len(items) > 0:
            test[u] = items.tolist()

    return test