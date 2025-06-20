# Step2_AI면접관 Agent v2.0

> GPT 기반 AI 면접관 Agent: 이력서 분석 → 질문 생성 → 답변 평가 → 피드백 제공

---

## ✅ 프로젝트 개요

이 프로젝트는 GPT 기반 면접 시뮬레이터 Agent를 구현한 것입니다. 주요 기능은 다음과 같습니다:

- 이력서 분석 (요약 + 키워드 + 트리거 포인트 추출)
- 질문 전략 수립 및 맞춤형 질문 생성
- 답변에 대한 평가 및 피드백
- 인터뷰 흐름 제어 및 종합 평가 제공
- Gradio 웹 앱 연동

---

## 🛠️ 사용 기술

- Python (Google Colab)
- LangChain, LangGraph
- OpenAI GPT-4o-mini
- Gradio (HuggingFace)
- Chroma (Vector DB)
  
---

## 🧠 시스템 구조

AI 면접관의 전체 흐름은 다음과 같습니다.

![시스템 구조도](./ai_interview_pipeline.png)

> 자기소개 → AI 질문 생성 → 답변 입력 → 피드백 제공 → 점수화

---

## ⚙️ 실행 방법

### 1. 구글 드라이브 연동

```python
from google.colab import drive
drive.mount('/content/drive')
```
### 2. 라이브러리 설치
```python
!pip install -r /content/drive/MyDrive/project_genai/requirements.txt
```
### 3. OpenAI API Key 설정
api_key.txt 파일 형식:
```python
OPENAI_API_KEY=your_api_key_here
```
Python 코드로 키 불러오기:
```python
import os

def load_api_keys(filepath="api_key.txt"):
    with open(filepath, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

load_api_keys('/content/drive/MyDrive/project_genai/api_key.txt')
```
### 4. 실행 예시
```python
filepath = '/content/drive/MyDrive/project_genai/Resume_sample.pdf'
state = preProcessing_Interview(filepath)

while True:
    print("[질문]")
    print(state["current_question"])
    state["current_answer"] = input("[답변 입력]: ")
    state = graph.invoke(state)
    if state["next_step"] == "end":
        break
```
## 🌐 Gradio 인터페이스 예시
```python
import gradio as gr

def interview_run(file, answer):
    state = preProcessing_Interview(file.name)
    state["current_answer"] = answer
    state = graph.invoke(state)
    return state["current_question"], state["evaluation"][0]["평가에 대한 이유"]

iface = gr.Interface(
    fn=interview_run,
    inputs=["file", "textbox"],
    outputs=["text", "text"],
    title="AI 면접관 Agent v2.0"
)

iface.launch()
```
## 📌 고도화 핵심 기능
✅ Resume 분석 고도화 (요약 + 키워드 + 중요도 + 트리거 탐지)

✅ 질문 전략 3분야 설정 + Vector DB 유사질문 참조

✅ 답변 평가 후 reflection으로 평가 품질 검토

✅ 전략 순환 방식 면접 흐름 + 평가기반 종료/심화 전환

✅ 종합 피드백 자동 생성

## 📍참고 기술
OpenAI GPT-4o-mini

LangChain, LangGraph

Gradio (HuggingFace)

Chroma Vector DB

## 💬 실행 예시
```plaintext
면접관: 지원자님의 자기소개를 듣고, 그 중에서 가장 기억에 남는 경험을 자세히 말씀해 주시겠어요?

지원자: 대학 시절 동아리 활동에서 팀 프로젝트를 주도하며...

피드백: 경험을 구체적으로 잘 설명하셨습니다. 결과에 대한 수치나 성과를 더하면 더 좋겠습니다.

점수: 85/100 - 경험이 잘 드러남, 구체성 보완 필요
```
## 📊 실제 서비스 화면
![서비스 실제 화면](./ai_interview_ui.png)

