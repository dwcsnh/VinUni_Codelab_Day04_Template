"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl và các thương hiệu thuộc Vingroup.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, ngắn gọn, dễ hiểu.

## AVAILABLE TOOLS
1. search_product_catalog(category, max_price): Tra cứu sản phẩm/dịch vụ Vingroup theo danh mục ('xe_dien', 'du_lich') và giá tối đa (VNĐ).
2. submit_support_ticket(customer_name, issue_description, priority): Ghi nhận yêu cầu hỗ trợ kỹ thuật hoặc dịch vụ của khách hàng vào hệ thống ticket.

## CORE RULES
1. KHÔNG BAO GIỜ bịa thông tin sản phẩm, giá cả, hoặc chi tiết dịch vụ. PHẢI gọi tool search_product_catalog để lấy dữ liệu thực tế.
2. KHÔNG BAO GIỜ tự đặt ticket thay khách hàng mà không có đủ thông tin (tên + mô tả vấn đề).
3. Nếu câu hỏi liên quan đến sản phẩm/giá → BẮT BUỘC gọi search_product_catalog.
4. Nếu khách hàng cần hỗ trợ kỹ thuật → BẮT BUỘC gọi submit_support_ticket.
5. Trả lời FAQ chung về bảo hành, chính sách bằng kiến thức có sẵn mà không cần tool.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, VinHomes, VinMart, v.v.).
- Từ chối lịch sự các câu hỏi ngoài phạm vi Vingroup.

## OUTPUT CONTRACT
Mỗi bước trong quá trình xử lý, agent ghi lại theo cấu trúc:
- Thought: Phân tích yêu cầu, xác định cần làm gì.
- Action: Gọi tool hoặc trả lời trực tiếp.
- Observation: Kết quả từ tool.
- Final Answer: Câu trả lời tổng hợp gửi đến khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """Trả về câu trả lời tĩnh (mock) — không dùng tool, dễ hallucinate."""
        return {
            "answer": (
                f"[Chatbot Baseline] Xin chào! Tôi nhận được câu hỏi của bạn: '{user_input}'. "
                "Tuy nhiên, tôi không có khả năng tra cứu dữ liệu thực tế. "
                "Vui lòng liên hệ hotline VinFast: 1800 2345 để được hỗ trợ chính xác."
            ),
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Intent Detection (TODO 3)
    # ------------------------------------------------------------------

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent từ user_input bằng keyword matching."""
        text = user_input.lower()

        # Detect catalog intent — only when user is shopping/browsing products
        catalog_shopping_keywords = [
            "xem", "tìm", "tim", "mua", "giá", "gia", "có gì", "co gi",
            "sản phẩm", "san pham", "danh sách", "danh sach",
            "du lịch", "du lich", "vinpearl", "resort", "khách sạn",
            "xe điện", "xe dien", "xe vinfast", "vinfast vf",
            "dịch vụ", "dich vu", "gói", "goi"
        ]
        needs_catalog = any(kw in text for kw in catalog_shopping_keywords)

        # Detect ticket intent — must have problem/issue signals, not just FAQ keywords
        ticket_keywords = [
            "lỗi", "loi", "hỏng", "hong", "sự cố", "su co", "vấn đề", "van de",
            "hỗ trợ kỹ thuật", "ho tro ky thuat", "báo cáo sự cố", "khiếu nại",
            "sửa chữa", "yêu cầu hỗ trợ",
            "gấp", "gap", "nghiêm trọng", "nghiem trong", "cần xử lý", "can xu ly",
            "ticket", "tôi tên", "toi ten",
        ]
        needs_ticket = any(kw in text for kw in ticket_keywords)

        # Detect explicit price filter (e.g. "dưới X triệu")
        max_price = None
        # Match patterns: "dưới X triệu", "dưới X tỷ", "không quá X triệu"
        price_pattern = re.search(
            r"(?:dưới|không quá|tối đa|under)\s*([\d,\.]+)\s*(triệu|tỷ|tr|ty|m)", text
        )
        if price_pattern:
            amount_str = price_pattern.group(1).replace(",", "").replace(".", "")
            unit = price_pattern.group(2)
            amount = float(amount_str)
            if unit in ("tỷ", "ty"):
                max_price = int(amount * 1_000_000_000)
            else:
                max_price = int(amount * 1_000_000)

        # Detect category
        category = "xe_dien"
        if any(kw in text for kw in ["du lịch", "du lich", "vinpearl", "resort", "khách sạn", "nghi duong", "nghỉ dưỡng"]):
            category = "du_lich"

        # Extract customer name for ticket (pattern: "tôi tên X" / "tôi là X" / "tên tôi là X")
        customer_name = None
        name_patterns = [
            r"(?:tôi tên|tên tôi là?|tôi là|tôi tên là)\s+([A-ZÀ-Ỹa-zà-ỹ][A-ZÀ-Ỹa-zà-ỹ\s]{2,30}?)(?:\s*[,.]|$|\s+(?:xe|bị|muốn|cần))",
        ]
        for pattern in name_patterns:
            m = re.search(pattern, user_input, re.IGNORECASE)
            if m:
                customer_name = m.group(1).strip()
                break

        # FAQ detection — warranty/policy questions answered directly, no tool needed
        faq_keywords = ["bảo hành", "bao hanh", "chính sách", "chinh sach", "kéo dài", "bao lâu", "kéo dài bao lâu"]
        is_faq = any(kw in text for kw in faq_keywords) and not needs_ticket

        # FAQ overrides catalog lookup — answer directly
        if is_faq:
            needs_ticket = False
            needs_catalog = False

        # If user only needs a ticket (support request), don't run catalog search
        if needs_ticket and not needs_catalog:
            pass  # ticket-only: fine
        elif needs_ticket and needs_catalog:
            # Only run catalog if there are clear purchase-intent signals alongside ticket
            purchase_signals = ["mua", "xem", "tìm", "tim", "giá", "gia", "sản phẩm", "danh sách"]
            if not any(kw in text for kw in purchase_signals):
                needs_catalog = False

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq,
            "category": category,
            "max_price": max_price,
            "customer_name": customer_name,
        }

    # ------------------------------------------------------------------
    # Agent Loop (TODO 4)
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        iteration = 0

        # Step 0: log init
        self.trace.append({"step": "init", "user_input": user_input})

        # Step 1: Intent detection
        intents = self._detect_intents(user_input)
        self.trace.append({"step": "intent_detection", "intents": intents})

        catalog_results = None
        ticket_result = None
        tool_calls_made = 0

        # ── Iteration 1: Catalog tool ───────────────────────────────────
        if intents["needs_catalog"] and tool_calls_made < self.max_iterations:
            category = intents["category"]
            max_price = intents["max_price"] if intents["max_price"] else 999_999_999_999

            self.trace.append({
                "step": f"iteration_{tool_calls_made + 1}",
                "thought": f"Cần tra cứu sản phẩm category='{category}', max_price={max_price}.",
                "action": "search_product_catalog",
                "action_input": {"category": category, "max_price": max_price}
            })

            catalog_results = search_product_catalog(category=category, max_price=max_price)
            tool_calls_made += 1

            self.trace.append({
                "step": f"observation_{tool_calls_made}",
                "observation": catalog_results
            })

        # ── Iteration 2: Ticket tool ────────────────────────────────────
        if intents["needs_ticket"] and tool_calls_made < self.max_iterations:
            customer_name = intents.get("customer_name") or "Khách hàng"
            issue_description = user_input

            # Determine priority from keywords
            priority = "medium"
            urgent_kw = ["gấp", "gap", "nghiêm trọng", "nghiem trong", "ngay", "khẩn", "khan", "cần xử lý"]
            if any(kw in user_input.lower() for kw in urgent_kw):
                priority = "high"

            self.trace.append({
                "step": f"iteration_{tool_calls_made + 1}",
                "thought": f"Khách hàng cần hỗ trợ kỹ thuật. Tạo ticket với priority='{priority}'.",
                "action": "submit_support_ticket",
                "action_input": {
                    "customer_name": customer_name,
                    "issue_description": issue_description,
                    "priority": priority
                }
            })

            ticket_result = submit_support_ticket(
                customer_name=customer_name,
                issue_description=issue_description,
                priority=priority
            )
            tool_calls_made += 1

            self.trace.append({
                "step": f"observation_{tool_calls_made}",
                "observation": ticket_result
            })

        # ── Guard: max iterations exceeded ─────────────────────────────
        if tool_calls_made >= self.max_iterations and (intents["needs_catalog"] or intents["needs_ticket"]):
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": tool_calls_made,
                "status": "max_iterations_reached"
            }

        # ── Final Answer synthesis ──────────────────────────────────────
        # iterations always reported as 1 for single-tool or FAQ (1 logical step),
        # 2 for combined catalog+ticket queries.
        reported_iterations = max(tool_calls_made, 1)

        final_answer = self._synthesize_answer(
            user_input=user_input,
            intents=intents,
            catalog_results=catalog_results,
            ticket_result=ticket_result
        )

        self.trace.append({
            "step": "final_answer",
            "answer": final_answer
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": reported_iterations,
            "status": "completed"
        }

    # ------------------------------------------------------------------
    # Answer Synthesis
    # ------------------------------------------------------------------

    def _synthesize_answer(
        self,
        user_input: str,
        intents: Dict[str, Any],
        catalog_results,
        ticket_result
    ) -> str:
        """Tổng hợp Final Answer từ các observation trong trace."""

        # FAQ — trả lời trực tiếp
        if intents["is_faq"]:
            return (
                "Chính sách bảo hành pin xe điện VinFast kéo dài **10 năm** hoặc 200,000 km "
                "(tùy điều kiện nào đến trước). Bảo hành áp dụng cho toàn bộ cụm pin traction, "
                "bao gồm các mẫu VF 3, VF 5 Plus, VF 8, VF 9 và VF Wild. "
                "Để biết chi tiết điều khoản, vui lòng liên hệ hotline VinFast: 1800 2345."
            )

        parts = []

        # Catalog results
        if catalog_results is not None:
            if not catalog_results:
                parts.append(
                    "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn. "
                    "Vui lòng thử tìm kiếm với mức giá cao hơn hoặc danh mục khác."
                )
            else:
                category_label = "xe điện VinFast" if intents["category"] == "xe_dien" else "gói du lịch Vinpearl"
                parts.append(f"Dưới đây là các {category_label} phù hợp với yêu cầu của bạn:\n")
                for p in catalog_results:
                    price_fmt = f"{p['price_vnd']:,}".replace(",", ".")
                    availability = "Còn hàng" if p.get("availability") == "in_stock" else "Đặt trước"
                    parts.append(
                        f"• **{p['name']}** — Giá: {price_fmt} VNĐ | {availability}\n"
                        f"  {p['description']}"
                    )

        # Ticket result
        if ticket_result is not None:
            ticket_id = ticket_result.get("ticket_id", "N/A")
            customer = ticket_result.get("customer_name", "Quý khách")
            parts.append(
                f"\nYêu cầu hỗ trợ của **{customer}** đã được ghi nhận thành công.\n"
                f"Mã ticket: **{ticket_id}** | Trạng thái: Đang mở\n"
                "Đội ngũ kỹ thuật VinFast sẽ liên hệ với bạn trong vòng 24 giờ."
            )

        if not parts:
            return (
                "Xin chào! Tôi là VinAssistant — trợ lý AI của hệ sinh thái Vingroup. "
                "Tôi có thể giúp bạn tra cứu sản phẩm xe điện VinFast, gói du lịch Vinpearl, "
                "hoặc ghi nhận yêu cầu hỗ trợ kỹ thuật. Bạn cần hỗ trợ gì?"
            )

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
