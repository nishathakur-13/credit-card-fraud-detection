"""FraudShield's lightweight in-app AI assistant."""

import re
from html import escape

import streamlit as st


def _risk_label(probability):
	if probability >= 0.8:
		return "HIGH"
	if probability >= 0.5:
		return "MEDIUM"
	return "LOW"


def _answer(prompt, context):
	"""Answer common operational questions using the current dashboard state."""
	question = prompt.lower().strip()
	dataset = context.get("dataset")
	model = context.get("model")
	metrics = context.get("metrics") or {}
	result = context.get("result")

	if re.search(r"\b(hello|hi|hey)\b", question):
		return "Hello. I can explain the model, summarize fraud activity, or help interpret a transaction result."
	if any(phrase in question for phrase in ("help", "what can", "how do", "what does this", "what is this")):
		return "Try asking: 'How many fraud cases are there?', 'How does the model work?', or 'Explain my latest result.'"
	if "model" in question or "algorithm" in question or "random forest" in question:
		if model is None:
			return "The Random Forest model artifact is not loaded, so predictions are unavailable."
		features = getattr(model, "n_features_in_", "the configured")
		return f"FraudShield is using a loaded Random Forest model with {features} input features. It returns a fraud probability and a risk band."
	if any(word in question for word in ("metric", "accuracy", "precision", "recall", "f1", "performance")):
		if not metrics:
			return "Model metrics are not available until the dataset and model are loaded."
		return (f"Current evaluation: accuracy {metrics.get('accuracy', 0) * 100:.2f}%, "
				f"precision {metrics.get('precision', 0) * 100:.2f}%, "
				f"recall {metrics.get('recall', 0) * 100:.2f}%, and "
				f"F1 {metrics.get('f1', 0) * 100:.2f}%.")
	if any(word in question for word in ("fraud", "alert", "transaction", "case", "count")):
		if dataset is None or dataset.empty or "Class" not in dataset:
			return "The transaction dataset is unavailable, so I cannot summarize fraud activity."
		fraud_count = int(dataset["Class"].sum())
		fraud_rate = fraud_count / len(dataset) * 100
		return f"The dataset contains {len(dataset):,} transactions, including {fraud_count:,} labeled fraud cases ({fraud_rate:.3f}%)."
	if re.search(r"(threshold|risk level|high risk|medium risk|low risk)", question):
		return "Risk bands are LOW below 50%, MEDIUM from 50% to below 80%, and HIGH at 80% or above."
	if any(word in question for word in ("result", "risk", "prediction", "decision")):
		if not result:
			return "There is no latest transaction result yet. Open Analyze Transaction and run an analysis first."
		prediction, probability = result
		decision = {"HIGH": "BLOCK", "MEDIUM": "REVIEW", "LOW": "APPROVE"}[_risk_label(probability)]
		return f"The latest result is {_risk_label(probability)} risk at {probability * 100:.2f}% fraud probability. Recommended action: {decision}."
	return "I can help with fraud counts, model performance, risk thresholds, or your latest transaction result."


def render_chatbot(context):
	"""Render the assistant in the sidebar using the supplied live app context."""
	if "assistant_messages" not in st.session_state:
		st.session_state.assistant_messages = [
			{"role": "assistant", "content": "I am your FraudShield assistant. Ask me about your data or model."}
		]

	st.markdown("""
		<div class="assistant-panel">
			<div class="assistant-heading"><div class="assistant-avatar">✦</div><div><div class="assistant-name">AI Fraud Assistant</div><div class="assistant-status"><span></span> Online · Ready to help</div></div><div class="assistant-spark">✧</div></div>
			<div class="assistant-rule"></div>
		</div>
	""", unsafe_allow_html=True)
	with st.expander("Chat with assistant", expanded=False):
		st.markdown('<div class="assistant-prompt-label">QUICK QUESTIONS</div>', unsafe_allow_html=True)
		quick_questions = ["How many fraud cases?", "Show model metrics", "Explain risk levels"]
		quick_columns = st.columns(3)
		for index, question in enumerate(quick_questions):
			if quick_columns[index].button(question, key=f"assistant_quick_{index}"):
				st.session_state.assistant_messages.append({"role": "user", "content": question})
				st.session_state.assistant_messages.append({"role": "assistant", "content": _answer(question, context)})
				st.rerun()
		for message in st.session_state.assistant_messages:
				bubble_class = "assistant-bubble user-bubble" if message["role"] == "user" else "assistant-bubble"
				content = escape(message["content"])
				st.markdown(f'<div class="{bubble_class}">{content}</div>', unsafe_allow_html=True)
		prompt = st.chat_input("Ask about fraud risk...")
		if prompt:
			st.session_state.assistant_messages.append({"role": "user", "content": prompt})
			st.session_state.assistant_messages.append({"role": "assistant", "content": _answer(prompt, context)})
			st.rerun()
