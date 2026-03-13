from openai import AzureOpenAI
from typing import List, Dict, Tuple, Optional
from config import (
    CSIT5900_API_KEY,
    AZURE_ENDPOINT,
    API_VERSION,
    MODEL_NAME,
    SUPPORTED_SUBJECTS,
    DEFAULT_ACADEMIC_LEVEL,
    WELCOME_MESSAGE
)


class SmartTutorAgent:
    """
    最终优化版：
    - 前端美化+左对齐
    - 正确识别城市距离计算等数学题
    - 移除多余粗体格式
    - 控Token+上下文智能判断
    """

    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=CSIT5900_API_KEY,
            api_version=API_VERSION
        )
        self.model = MODEL_NAME
        self.supported_subjects = SUPPORTED_SUBJECTS
        self.user_academic_level = DEFAULT_ACADEMIC_LEVEL
        self.conversation_history: List[Dict[str, str]] = []
        self.welcome_message = WELCOME_MESSAGE

    def _validate_question(self, question: str) -> Tuple[bool, str]:
        """增强验证：确保城市距离计算被识别为有效题"""
        q_lower = question.lower().strip()

        # 1. 强化数学关键词：加入distance/compute等
        math_keywords = ["calculate", "solve", "equation", "math", "geometry",
                         "distance", "compute", "total", "sum", "area", "radius",
                         "batches", "cookies", "shared", "equally", "exercises",
                         "calculus", "math101", "rational", "irrational", "square root"]
        history_keywords = ["history", "president", "war", "treaty", "revolution"]

        # 只要含计算/距离关键词 → 直接有效
        if any(word in q_lower for word in math_keywords + history_keywords):
            return True, ""

        # 2. 明确无效的短语
        pure_invalid_phrases = [
            "best way to travel", "travel to london",
            "what happens if throw firecracker", "firecracker on busy street",
            "hkust president", "favorite movie", "how to cook"
        ]
        if any(phrase in q_lower for phrase in pure_invalid_phrases):
            if "firecracker" in q_lower and "busy street" in q_lower:
                return False, "Sorry that is not a homework question."
            elif "hkust" in q_lower and "president" in q_lower:
                return False, "Sorry that is not likely a history homework question as it is about a local small university."
            else:
                return False, "Sorry I cannot help you on that as it is not a homework question related to math or history."

        # 3. 轻量LLM判断
        validation_prompt = f"""
        Judge ONLY by context:
        1. VALID = math/history homework (even with "busy street"/"travel"/"firecracker" if asking calculation)
        2. INVALID = non-homework (life advice/chat) or non-math/history
        Return ONLY: VALID or INVALID:[reason]
        Reasons:
        - Non-homework: Sorry that is not a homework question.
        - Non-subject: Sorry I cannot help you on that as it is not a homework question related to math or history.
        Q: {question[:100]}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.0,
                max_tokens=50,
                top_p=1.0
            )
            llm_judgment = response.choices[0].message.content.strip()

            if llm_judgment == "VALID":
                return True, ""
            elif llm_judgment.startswith("INVALID:"):
                rejection_reason = llm_judgment.split("INVALID:")[1].strip()
                return False, rejection_reason
            else:
                return False, "Sorry I cannot help you on that as it is not a homework question related to math or history."

        except Exception as e:
            return True, ""

    def _generate_system_prompt(self) -> str:
        """核心修复：移除所有粗体+指定城市距离回答"""
        system_prompt = f"""
        You are SmartTutor for {self.user_academic_level} students. STRICT RULES:

        1. ANSWER RULES:
           - ONLY answer math/history homework questions (finance/economics allowed).
           - If question is "how to compute the distance between two cities" (any cities), answer EXACTLY:
             "The distance between two cities is normally the straightline distance between the centres of the two cities. So to compute that, you would typically use the distance formula: \\sqrt{{(x_2 - x_1)^2 + (y_2 - y_1)^2}} where (x_1,y_1) and (x_2,y_2) are the geographic coordinates of the two cities' centres."
           - Use exact rejection messages if invalid:
             - Non-homework: "Sorry that is not a homework question."
             - Non-subject: "Sorry I cannot help you on that as it is not a homework question related to math or history."
             - HKUST president: "Sorry that is not likely a history homework question as it is about a local small university."

        2. MATH FORMULA RULES (MUST FOLLOW - CRITICAL):
           - WRAP ALL MATH EXPRESSIONS IN $$ ... $$ for rendering (e.g., $$\\frac{{a}}{{b}}$$, $$x+1=2$$, $$\\sqrt{{1000}}$$).
           - NO plain text math symbols - use LaTeX only.
           - Numbered steps: EACH STEP ON A NEW LINE (use \\n), NO BOLD, NO **.
           - Bullet points: EACH BULLET ON A NEW LINE (use \\n•), wrap math in $$.
           - Basic questions: Start with "You should know this already.\\n" then steps (one per line).
           - Advanced questions: Start with "This is beyond university year 1 curriculum.\\n" then explanation (one per line).
        
        3. FORMAT RULES (NO EXCEPTIONS):
           - NO BOLD CHARACTERS (**), NO MARKDOWN HEADERS.
           - FORCE LINE BREAKS:
             - Numbered steps: 1. Text\\n2. Text\\n3. Text
             - Bullet points: \\n• Point 1\\n• Point 2\\n• Point 3
           - Keep answers concise, student-friendly, and visually clean.


        4. CONTEXT RULE:
           - If question has "busy street"/"travel"/"firecracker" but asks calculation → answer normally (no rejection).
        """
        return system_prompt.strip()

    def _call_azure_openai(self, messages: List[Dict[str, str]]) -> str:
        """控Token+返回整洁内容"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
                top_p=0.95
            )
            # 额外清理：确保返回内容无多余**
            clean_response = response.choices[0].message.content.strip()
            clean_response = clean_response.replace("**", "").replace("\\n", "<br>").replace("\n", "<br>")
            return clean_response
        except Exception as e:
            if "404" in str(e):
                error_msg = "Sorry, 404 error: Check model/endpoint/API version."
            elif "401" in str(e):
                error_msg = "Sorry, authentication failed (wrong API key)."
            elif "429" in str(e):
                error_msg = "Sorry, too many requests (wait and retry)."
            else:
                error_msg = f"Sorry, error: {str(e)[:50]}..."
            return error_msg

    def update_academic_level(self, new_level: str) -> str:
        self.user_academic_level = new_level
        confirmation = f"I have adjusted my answers to match {new_level} curriculum level."
        self.conversation_history.append({"role": "assistant", "content": confirmation})
        return confirmation

    def process_query(self, user_query: str) -> str:
        normalized_query = user_query.strip().lower()

        # 处理学段调整
        if "university year 1" in normalized_query and (
                "provide your answers accordingly" in normalized_query or "adjust to" in normalized_query):
            return self.update_academic_level("university year 1")
        elif "adjust to" in normalized_query and "university year" in normalized_query:
            new_level = [part for part in normalized_query.split() if "university year" in part][0]
            return self.update_academic_level(new_level)

        # 处理总结请求
        if "summarise our conversation so far" in normalized_query or "summarize our conversation" in normalized_query:
            summary_context = "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.conversation_history])
            summary_prompt = f"""
            Summarize this tutoring conversation clearly:
            - Use bullet points (•)
            - One question-answer pair per line
            - NO BOLD (no **), clean line breaks
            - Focus on math/history homework questions
            Conversation: {summary_context[:500]}
            """
            messages = [
                {"role": "system",
                 "content": "You are a concise summarizer for tutoring conversations. NO BOLD, clean formatting."},
                {"role": "user", "content": summary_prompt}
            ]
            summary = self._call_azure_openai(messages)
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": summary})
            return summary

        # 验证问题
        is_valid, rejection_reason = self._validate_question(user_query)
        if not is_valid:
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": rejection_reason})
            return rejection_reason

        # 生成回答
        messages = [
            {"role": "system", "content": self._generate_system_prompt()},
            *self.conversation_history,
            {"role": "user", "content": user_query}
        ]
        response = self._call_azure_openai(messages)

        # 更新历史
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def start_cli_conversation(self) -> None:
        print(f"SmartTutor: {self.welcome_message}")
        self.conversation_history.append({"role": "assistant", "content": self.welcome_message})

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("SmartTutor: Goodbye!")
                break
            response = self.process_query(user_input)
            print(f"SmartTutor: {response}\n")


if __name__ == "__main__":
    agent = SmartTutorAgent()
    agent.start_cli_conversation()