import numpy as np
import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 사용자-아이템 평점 행렬 생성
user_item_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating')

# 결측치는 0으로 채우기 (평가되지 않은 영화는 중립적으로 간주)
user_item_matrix.fillna(0, inplace=True)

# 잠재 요인의 개수
n_factors = 5

# 사용자와 아이템의 잠재 요인 행렬 초기화
user_factors = np.random.normal(scale=1./n_factors, size=(user_item_matrix.shape[0], n_factors))
item_factors = np.random.normal(scale=1./n_factors, size=(user_item_matrix.shape[1], n_factors))


def matrix_factorization(R, P, Q, steps=100, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            idx = np.where(R[i] > 0)[0]
            Ri = R[i, idx]
            Pi = P[i, :]
            Qi = Q[:, idx]
            Ei = Ri - np.dot(Pi, Qi)

            P[i, :] += alpha * (np.dot(Ei, Qi.T) - beta * Pi)
            for j in range(len(idx)):
                Q[:, idx[j]] += alpha * (Pi * Ei[j] - beta * Q[:, idx[j]])

        eR = np.dot(P, Q)
        e = np.sum((R[R > 0] - eR[R > 0]) ** 2) + beta / 2. * (np.sum(P ** 2) + np.sum(Q ** 2))
        if e < 0.001:
            break

    return P, Q.T

user_factors, item_factors = matrix_factorization(user_item_matrix.values, user_factors, item_factors, steps=100, alpha=0.0002, beta=0.02)

def predict_rating(user_id, item_id):
    user_idx = list(user_item_matrix.index).index(user_id)
    item_idx = list(user_item_matrix.columns).index(item_id)
    return np.dot(user_factors[user_idx], item_factors.T[item_idx])

test_data['predicted_rating'] = test_data.apply(lambda row: predict_rating(row['userId'], row['movieId']), axis=1)

submission = test_data[['rId', 'predicted_rating']]
submission.rename(columns={'predicted_rating': 'rating'}, inplace=True)
submission.to_csv('submission2.csv', index=False)
