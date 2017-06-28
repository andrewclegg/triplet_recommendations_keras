import numpy as np

from sklearn.metrics import roc_auc_score


def full_auc(model, ground_truth, item_features):
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    ground_truth = ground_truth.tocsr()

    no_users, no_items = ground_truth.shape

    pid_array = np.arange(no_items, dtype=np.int32)

    test_item_features = item_features[pid_array, :]

    scores = []

    for user_id, row in enumerate(ground_truth):

        predictions = model.predict({
            'positive_item_input': pid_array,
            'positive_tags': test_item_features,
            'user_input': np.repeat(user_id, no_items)},
            batch_size=no_items)

        true_pids = row.indices[row.data == 1]

        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1

        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))

    return sum(scores) / len(scores)
