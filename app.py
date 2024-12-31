import numpy as np
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import torch
import uvicorn

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("./KR-SBERT-V40K-klueNLI-augSTS")
model = AutoModel.from_pretrained("./KR-SBERT-V40K-klueNLI-augSTS")



# JSON 변환 함수
def convert_to_python(data):
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data

def encode_texts(texts):
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

# def load_model():
#     """
#     모델과 토크나이저를 로드합니다.
#     """
#     try:
#         print("Loading model...")
#         tokenizer = AutoTokenizer.from_pretrained("./KR-SBERT-V40K-klueNLI-augSTS")
#         model = AutoModel.from_pretrained("./KR-SBERT-V40K-klueNLI-augSTS")
#         print("Model loaded successfully")
#         return tokenizer, model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Please ensure the model name is correct or available in the Hugging Face repository.")
#         return None, None

# # 모델 및 토크나이저 로드
# tokenizer, model = load_model()

# def encode_texts(texts):
#     """
#     텍스트 데이터를 SBERT 임베딩으로 변환합니다.
#     """
#     if not tokenizer or not model:
#         raise ValueError("Model or tokenizer not loaded")
    
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#     with torch.no_grad():
#         model_output = model(**inputs)
#     embeddings = model_output.last_hidden_state.mean(dim=1)
#     return embeddings

@app.post("/similarity")
async def similarity(request: Request):
    data = await request.json()
    questions = data["questions"]

    texts = [q["text"] for q in questions]
    answers = [q["answer"] for q in questions]

    # 1. 텍스트 임베딩 생성
    embeddings = encode_texts(texts)

    # 2. 코사인 유사도 계산
    similarity_matrix = cosine_similarity(embeddings)

    # 3. 유사도를 거리로 변환
    distance_matrix = 1 - similarity_matrix
    distance_matrix[distance_matrix < 0] = 0  # 음수 값 방지

    # 4. DBSCAN 클러스터링
    clustering = DBSCAN(eps=0.5, min_samples=2, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    # 5. 클러스터 결과 생성
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({"text": texts[idx], "answer": answers[idx]})

    # return {"clusters": clusters}
    # 6. JSON 변환
    response = {
        convert_to_python(key): [convert_to_python(item) for item in value]
        for key, value in clusters.items()
    }
    return response

# 중복 확인
@app.post("/check_similarity")
async def check_similarity(request: Request):
    data = await request.json()

    # 새로운 질문과 기존 질문들
    new_question = data["new_question"]
    existing_questions = data["existing_questions"]

    # 1. 텍스트 임베딩 생성
    all_texts = [new_question] + existing_questions
    embeddings = encode_texts(all_texts)

    # 2. 코사인 유사도 계산
    similarity_scores = cosine_similarity(embeddings[0:1], embeddings[1:])[0]

    # 3. 중복 여부 판단
    threshold = 0.8
    is_duplicate = any(score >= threshold for score in similarity_scores)

    # 4. 결과 반환
    return {
        "is_duplicate": is_duplicate,
        "similarity_scores": similarity_scores.tolist(),
        "threshold": threshold,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)
