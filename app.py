import numpy as np
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import torch
import uvicorn
import openai
import os

# 발급받은 API 키 설정
OPENAI_API_KEY = "API_KEY"

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

app = FastAPI()


tokenizer = AutoTokenizer.from_pretrained("./KR-SBERT-V40K-klueNLI-augSTS")
model = AutoModel.from_pretrained("./KR-SBERT-V40K-klueNLI-augSTS")

# GPT 모델 설정
gpt_model = "gpt-3.5-turbo"

def encode_texts(texts):
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

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

    # LLM을 사용하여 새로운 질문/답변 생성
    generated_faqs = await generate_faqs_with_llm(clusters, existing_questions=data.get("existing_questions", []))

    # 6. JSON 변환
    response = {
        "clusters": {
            convert_to_python(key): [convert_to_python(item) for item in value]
            for key, value in clusters.items()
        },
        "generated_faqs": generated_faqs,
    }
    return response

async def generate_faqs_with_llm(clusters, existing_questions):
    """
    LLM을 사용하여 클러스터 기반으로 새로운 질문과 답변을 생성합니다.
    기존 질문과의 중복 여부도 검사합니다.
    """
    generated_faqs = []
    threshold = 0.85 # 유사도 임계값

    # 기존 질문이 비어 있는 경우 바로 처리
    if not existing_questions:
        print("No existing questions provided. Skipping similarity check.")
        for cluster_id, cluster_items in clusters.items():
            cluster_texts = [item["text"] for item in cluster_items]
            cluster_answers = [item["answer"] for item in cluster_items]
            prompt = generate_llm_prompt(cluster_texts, cluster_answers)
            messages = [
                {"role": "system", "content": "You are an assistant that generates FAQs by clustering similar questions."},
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(model=gpt_model, messages=messages)
            generated_text = response['choices'][0]['message']['content'].strip()
            generated_question, generated_answer = parse_llm_response(generated_text)

            if generated_question and generated_answer:
                generated_faqs.append({"question": generated_question, "answer": generated_answer})

        return generated_faqs

    for cluster_id, cluster_items in clusters.items():
        cluster_texts = [item["text"] for item in cluster_items]
        cluster_answers = [item["answer"] for item in cluster_items]

        # LLM 요청에 사용할 프롬프트 생성
        prompt = generate_llm_prompt(cluster_texts, cluster_answers)

        # 메시지 설정하기
        messages = [
            {"role": "system", "content": "당신은 유사한 질문을 클러스터링하여 FAQ를 생성하는 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=gpt_model, messages=messages)
        generated_text = response['choices'][0]['message']['content'].strip()

        # LLM 응답 파싱
        generated_question, generated_answer = parse_llm_response(generated_text)

        # 중복 검사: 기존 질문들과의 코사인 유사도 계산
        all_texts = [generated_question] + existing_questions
        embeddings = encode_texts(all_texts)
        similarity_scores = cosine_similarity(embeddings[0:1], embeddings[1:])[0]

        if any(score >= threshold for score in similarity_scores):
            print(f"중복된 질문입니다 : \"{generated_question}\"")
            continue  # 중복인 경우 추가하지 않음

        generated_faqs.append({
            "question": generated_question,
            "answer": generated_answer
        })

    return generated_faqs


def generate_llm_prompt(questions, answers):
    """
    LLM에 사용할 프롬프트를 생성합니다.
    """
    prompt = (
        "다음은 분리수거 어플 사용자들이 질문한 다양한 질문과 관리자 답변을 유사도 분석으로 묶은 결과입니다. 이 목록을 바탕으로 하나의 통합된 질문과 답변을 생성하여 자주 묻는 질문(FAQ)으로 사용할 수 있도록 작성하세요.\n\n"
    )
    for i, (question, answer) in enumerate(zip(questions, answers)):
        prompt += f"질문 {i + 1}: {question}\n답변 {i + 1}: {answer}\n\n"

    prompt += "통합된 질문과 답변을 생성하세요:\n질문: "
    return prompt


def parse_llm_response(response_text):
    """
    LLM 응답에서 질문과 답변을 파싱합니다.
    """
    lines = response_text.split("\n")
    question = lines[0].replace("Q: ", "").strip()
    answer = lines[1][lines[1].find('답변: ') + len('답변: '):].strip() if '답변: ' in lines[1] else lines[1].strip()
    # answer = lines[1].replace("A: ", "").strip()
    return question, answer

# JSON 변환 함수ㄴ
def convert_to_python(data):
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5005)