# app.py
import os
import io
import random
import requests
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="AI-Ассистент для Событий (Премиум-версия)", layout="centered")

# --- Конфигурация Hugging Face из переменных окружения
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "YOUR_HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "deepface/insightface-emotion")  # placeholder, при необходимости замените

# --- UI: заголовок и описание
st.title("AI-Ассистент для Событий (Премиум-версия)")
st.markdown("Анализ эмоций гостей по фото + персональные идеи для повышения вовлечённости и дохода")

# --- Сайдбар: монетизация
with st.sidebar:
    st.header("Монетизация")
    st.write("- **Тестовый пакет:** 5к")
    st.write("- **Подписка:** 5к/мес")
    st.write("- **Upsell кастомизация:** 10к")
    st.write("Подписка экономит **~10 часов** подготовки на событии")

# --- Загрузка изображения и опрос
st.subheader("Загрузите фото гостя")
uploaded_file = st.file_uploader("JPG/PNG, до 5 МБ", type=["jpg", "jpeg", "png"])

st.subheader("Опрос гостя")
mood = st.slider("Настроение 1-5", min_value=1, max_value=5, value=3)

analyze = st.button("Анализировать")

# --- Вспомогательные функции
def call_hf_emotion_api(image_bytes: bytes, token: str, model: str, timeout=8):
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(api_url, headers=headers, data=image_bytes, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    # Ожидаем формат: [{'label': 'happy', 'score': 0.9}, ...] или {'label': score, ...}
    result = {}
    if isinstance(data, list):
        for item in data:
            label = item.get("label") or item.get("class") or str(item)
            score = item.get("score") or item.get("confidence") or 0
            result[label.lower()] = float(score)
    elif isinstance(data, dict):
        # Иногда HF возвращает словарь label: score
        if all(isinstance(v, (int, float)) for v in data.values()):
            for k, v in data.items():
                result[str(k).lower()] = float(v)
        else:
            # Попробуем преобразовать вложенный формат
            for k, v in data.items():
                try:
                    result[str(k).lower()] = float(v)
                except Exception:
                    pass
    # Нормализация
    s = sum(result.values()) or 1.0
    for k in list(result.keys()):
        result[k] = result[k] / s
    return result

def mock_emotion_detector(image_bytes: bytes):
    emotions = ["happy", "sad", "neutral", "angry", "surprised", "disgusted", "fearful"]
    chosen = random.choice(emotions)
    probs = {}
    for em in emotions:
        if em == chosen:
            probs[em] = 0.6 + random.random() * 0.35
        else:
            probs[em] = random.random() * 0.1
    s = sum(probs.values())
    for k in probs:
        probs[k] = probs[k] / s
    return probs

def pick_primary_emotion(probs: dict):
    if not probs:
        return "neutral", 1.0
    primary = max(probs.items(), key=lambda x: x[1])
    return primary[0], primary[1]

IDEA_TEMPLATES = {
    "happy": {
        "ideas": ["танцы на сцене с интерактивом", "мини-шоу или фейерверк"],
        "roi": "Это повысит отзывы на 20%, средний чек события +15%"
    },
    "sad": {
        "ideas": ["мини-игра с призами для поднятия настроения", "подбор музыкальных пауз и живой музыки"],
        "roi": "Это повысит отзывы на 20%, вовлечённость +18%"
    },
    "neutral": {
        "ideas": ["интерактив-опросы и голосования в реальном времени", "фотозона с короткими челленджами"],
        "roi": "Это повысит вовлечённость на 25%, отзывы +12%"
    },
    "angry": {
        "ideas": ["персонализированный ведущий/модерация", "быстрая развлекательная пауза"],
        "roi": "Снизит негативные отзывы на 30%, сохранит лояльность"
    },
    "surprised": {
        "ideas": ["момент-сюрприз от организаторов", "неожиданные подарки/акции"],
        "roi": "Увеличит вирусность фото/постов на 35%"
    },
    "disgusted": {
        "ideas": ["коррекция сценария и фуд-корта", "легкие развлечения для смены фокуса"],
        "roi": "Уменьшит отток гостей и негативные отзывы"
    },
    "fearful": {
        "ideas": ["спокойная зона и мягкая музыка", "безопасное взаимодействие с персоналом"],
        "roi": "Снизит стресс гостей, повысит комфорт и отзывы"
    }
}

def combine_with_poll(emotion: str, emotion_score: float, mood_value: int):
    if mood_value >= 4:
        poll_text = "опрошенный отметил высокое настроение"
    elif mood_value == 3:
        poll_text = "опрошенный отметил нейтральное настроение"
    else:
        poll_text = "опрошенный отметил низкое настроение"
    if emotion in ["happy", "surprised"]:
        base = "Гость выглядит позитивным"
    elif emotion in ["sad", "fearful", "disgusted", "angry"]:
        base = "Гость выглядит негативно/взволнованно"
    else:
        base = "Гость выглядит нейтрально"
    combined = f"{base} ({emotion} — {int(emotion_score*100)}%), {poll_text}."
    return combined

# --- Основная логика при нажатии Анализировать
if analyze:
    if not uploaded_file:
        st.error("Пожалуйста, загрузите фото перед нажатием Анализировать.")
    else:
        uploaded_file.seek(0, io.SEEK_END)
        size = uploaded_file.tell()
        uploaded_file.seek(0)
        max_bytes = 5 * 1024 * 1024
        if size > max_bytes:
            st.error("Файл слишком большой. Максимум 5 МБ.")
        else:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                img_thumb = ImageOps.fit(img, (400, 400))
                st.image(img_thumb, caption="Загруженное фото", use_column_width=False)
            except Exception:
                st.error("Не удалось открыть изображение. Загрузите корректный JPG/PNG.")
                st.stop()

            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

            probs = None
            if HF_API_TOKEN and HF_API_TOKEN != "YOUR_HF_API_TOKEN":
                try:
                    with st.spinner("Обращаюсь к Hugging Face для детекции эмоций..."):
                        probs = call_hf_emotion_api(image_bytes, HF_API_TOKEN, HF_MODEL)
                except Exception:
                    st.warning("Hugging Face API недоступен или вернул ошибку, использую мок-режим для демонстрации.")
                    probs = mock_emotion_detector(image_bytes)
            else:
                st.info("Токен Hugging Face не задан, использую мок-режим для теста.")
                probs = mock_emotion_detector(image_bytes)

            primary_emotion, primary_score = pick_primary_emotion(probs)

            st.subheader("Результат анализа")
            st.markdown(f"**Основная эмоция:** **{primary_emotion}** — **{int(primary_score*100)}% уверености**")

            st.write("Детализация вероятностей:")
            for k, v in sorted(probs.items(), key=lambda x: -x[1])[:6]:
                st.markdown(f"- **{k}**: {int(v*100)}%")

            combo_text = combine_with_poll(primary_emotion, primary_score, mood)
            st.info(combo_text)

            template_key = primary_emotion if primary_emotion in IDEA_TEMPLATES else "neutral"
            template = IDEA_TEMPLATES[template_key]
            st.subheader("Персональные идеи для вмешательства")
            st.write("Рекомендованные идеи:")
            for idea in template["ideas"][:3]:
                st.markdown(f"- **{idea}**")
            st.markdown(f"**Ожидаемый эффект:** {template['roi']}")

            st.subheader("Бизнес-оценка")
            st.markdown("- **ROI пример:** увеличит положительные отзывы на ~20%, средний чек +10–15% при правильной реализации.")
            st.markdown("- **Монетизация в продукте:** тест 5к; подписка 5к/мес; кастомные решения от 10к.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Сохранить рекомендацию в PDF (демо)"):
                    pdf_text = f"AI Ассистент — рекомендации\nЭмоция: {primary_emotion} ({int(primary_score*100)}%)\n\nИдеи:\n"
                    for i, idea in enumerate(template["ideas"], 1):
                        pdf_text += f"{i}. {idea}\n"
                    pdf_text += f"\nЭффект: {template['roi']}\n\nКомбо: {combo_text}\n"
                    st.success("Генерация демо-текста (в реальном продукте создайте PDF с reportlab/weasyprint).")
                    st.code(pdf_text)
            with col2:
                if st.button("Запросить демо у менеджера"):
                    st.success("Спасибо! Наш менеджер свяжется для демонстрации и обсуждения тарифов.")

# --- Футер
st.markdown("---")
st.markdown("**Подписка 5к/мес — сэкономь 10 часов на событии**")
st.caption("AI-Ассистент для Событий помогает анализировать эмоции гостей и предлагать точечные идеи для повышения вовлечённости и дохода")

# --- Комментарии по Stripe интеграции (коротко)
st.markdown("---")
st.subheader("Интеграция монетизации Stripe")
st.write("""
1. Создайте продукты в Stripe: тестовый пакет, подписка, кастом.
2. Используйте сервер (Flask/FastAPI) для создания Stripe Checkout session и webhook-обработки.
3. В Streamlit перенаправляйте пользователя на Checkout или открывайте ссылку.
4. Храните ключи в переменных окружения и обрабатывайте webhook'и для подтверждения платежей.
""")
st.caption("Рекомендуется отдельный бэкенд для безопасных вызовов Stripe и Hugging Face.")
