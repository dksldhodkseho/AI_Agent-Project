# 🤖 AI 면접관 Agent

자기소개를 입력하면 **AI가 실제 면접처럼 질문하고**,  
**답변을 평가하며 피드백과 점수를 제공하는 AI 면접 시스템**입니다.

---

## 🗂️ 프로젝트 개요

- **목표**:  
  사용자의 자기소개서를 입력받아, AI가 **심층 질문을 생성하고**,  
  사용자의 답변을 **자동 평가 및 점수화**, **개선 피드백**까지 제공합니다.

- **사용 기술**:  
  `Python`, `OpenAI GPT-3.5-turbo`, `Streamlit`

- **팀명**: AI 01반 2조  
- **팀원**: 최인규, 손정후, 김서영, 오승훈, 지용주, 조시현

---

## 🧠 시스템 구조

AI 면접관의 전체 흐름은 다음과 같습니다.

![시스템 구조도](./assets/ai_interview_pipeline.png)

> 자기소개 → AI 질문 생성 → 답변 입력 → 피드백 제공 → 점수화

---

## 🧩 핵심 기능 및 코드

각 기능을 개별 모듈로 구현하여 GPT API를 효과적으로 활용했습니다.

---

### 1️⃣ 면접 질문 생성 함수

```python
def generate_question(user_intro):
    prompt = f"아래 자기소개를 참고하여 지원자에게 던질 심층적인 면접 질문 한 가지를 만들어 주세요.\n\n자기소개: {user_intro}\n\n면접 질문:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 뛰어난 면접관입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()
```
✅ 사용자의 자기소개를 기반으로 AI가 적절한 면접 질문을 자동 생성합니다.

### 2️⃣ 답변 평가 및 피드백 함수
```python
def evaluate_answer(user_answer):
    prompt = f"지원자의 답변을 평가하고 개선점을 간단하게 제시해 주세요.\n\n답변: {user_answer}\n\n평가 및 피드백:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 뛰어난 면접관입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response.choices[0].message['content'].strip()
```
✅ AI가 답변을 분석하고, 개선점 중심의 피드백을 제공합니다.

### 3️⃣ 답변 점수화 함수
```python
def score_answer(user_answer):
    prompt = f"다음 답변에 대해 0~100점의 점수를 매기고, 한 줄로 이유를 설명해 주세요.\n\n답변: {user_answer}\n\n점수와 간단한 코멘트:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 뛰어난 면접관입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message['content'].strip()
```
✅ 점수와 함께 간단한 평가 코멘트를 제공합니다.

### 4️⃣ Streamlit 인터페이스
```python
import streamlit as st

st.title("AI 면접관 Agent")
user_intro = st.text_area("자기소개를 입력하세요.")
if st.button("면접 시작"):
    question = generate_question(user_intro)
    st.write("면접관:", question)

    user_answer = st.text_area("답변을 입력하세요.", key="answer1")
    if st.button("답변 제출"):
        feedback = evaluate_answer(user_answer)
        score = score_answer(user_answer)
        st.write("피드백:", feedback)
        st.write("점수:", score)
```
✅ 사용자와의 인터랙션을 위한 웹 UI 구성
✅ 면접 시작 → 질문 생성 → 답변 입력 → 피드백 및 점수 제공

## 💬 실행 예시
```plaintext
면접관: 지원자님의 자기소개를 듣고, 그 중에서 가장 기억에 남는 경험을 자세히 말씀해 주시겠어요?

지원자: 대학 시절 동아리 활동에서 팀 프로젝트를 주도하며...

피드백: 경험을 구체적으로 잘 설명하셨습니다. 결과에 대한 수치나 성과를 더하면 더 좋겠습니다.

점수: 85/100 - 경험이 잘 드러남, 구체성 보완 필요
## 📊 주요 결과
🔹 피드백 요약표

🔹 서비스 실제 화면

