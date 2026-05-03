"""Exportação de simulados em Markdown e PDF."""
from __future__ import annotations
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, KeepTogether, Flowable
)

from quiz import Question


# ---------- Markdown ----------

def to_markdown(
    questions: list[Question],
    answers: dict[str, int] | None = None,
    title: str = "Simulado CAMA",
    include_answers: bool = True,
) -> str:
    """
    Gera markdown do simulado.
    - Se `answers` for fornecido, marca a resposta do usuário.
    - Se `include_answers=False`, esconde gabarito (versão "prova em branco").
    """
    lines = [
        f"# {title}",
        f"_Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}_",
        f"",
        f"**Total de questões:** {len(questions)}",
        "",
        "---",
        "",
    ]

    if answers is not None:
        correct = sum(1 for q in questions if answers.get(q.id) == q.answer_idx)
        pct = correct / len(questions) * 100 if questions else 0
        lines += [
            f"## Resultado",
            f"- **Acertos:** {correct}/{len(questions)} ({pct:.1f}%)",
            "",
            "---",
            "",
        ]

    for i, q in enumerate(questions, 1):
        lines.append(f"## Questão {i}")
        lines.append(f"_Tópico: {q.topic}_")
        lines.append("")
        lines.append(q.stem)
        lines.append("")
        for j, opt in enumerate(q.options):
            letter = chr(65 + j)
            prefix = ""
            if include_answers and j == q.answer_idx:
                prefix = "✅ "
            if answers is not None:
                ua = answers.get(q.id)
                if j == ua and ua != q.answer_idx:
                    prefix = "❌ "
            lines.append(f"- {prefix}**{letter})** {opt}")
        lines.append("")

        if include_answers:
            lines.append(f"**Gabarito:** {chr(65 + q.answer_idx)}")
            lines.append("")
            lines.append(f"**Justificativa:** {q.explanation}")
            lines.append("")
            if q.sources:
                lines.append(f"**Fontes:** {', '.join(q.sources)}")
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ---------- PDF ----------

def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"],
            fontSize=18, leading=22, spaceAfter=12,
            textColor=HexColor("#1f4e3d"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"],
            fontSize=13, leading=16, spaceBefore=10, spaceAfter=6,
            textColor=HexColor("#1f4e3d"),
        ),
        "topic": ParagraphStyle(
            "topic", parent=base["Italic"],
            fontSize=9, leading=11, textColor=HexColor("#666666"),
            spaceAfter=6,
        ),
        "stem": ParagraphStyle(
            "stem", parent=base["BodyText"],
            fontSize=11, leading=14, spaceAfter=8, alignment=TA_LEFT,
        ),
        "option": ParagraphStyle(
            "option", parent=base["BodyText"],
            fontSize=10, leading=13, leftIndent=14, spaceAfter=2,
        ),
        "option_correct": ParagraphStyle(
            "option_correct", parent=base["BodyText"],
            fontSize=10, leading=13, leftIndent=14, spaceAfter=2,
            textColor=HexColor("#1a7a3a"), fontName="Helvetica-Bold",
        ),
        "option_wrong": ParagraphStyle(
            "option_wrong", parent=base["BodyText"],
            fontSize=10, leading=13, leftIndent=14, spaceAfter=2,
            textColor=HexColor("#b3261e"),
        ),
        "answer": ParagraphStyle(
            "answer", parent=base["BodyText"],
            fontSize=10, leading=13, spaceBefore=6, spaceAfter=2,
            fontName="Helvetica-Bold",
        ),
        "explanation": ParagraphStyle(
            "explanation", parent=base["BodyText"],
            fontSize=9, leading=12, spaceAfter=4,
            backColor=HexColor("#f4f4f0"), borderPadding=6,
            leftIndent=4, rightIndent=4,
        ),
        "sources": ParagraphStyle(
            "sources", parent=base["BodyText"],
            fontSize=8, leading=10, textColor=HexColor("#666666"),
            spaceAfter=8,
        ),
        "meta": ParagraphStyle(
            "meta", parent=base["BodyText"],
            fontSize=10, leading=13, spaceAfter=4,
        ),
    }


def _escape(text: str) -> str:
    """Escapa caracteres que o reportlab interpreta como markup."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

def _draw_footer(canvas, doc):
    """Rodapé desenhado em cada página do PDF.

    Layout em 2 linhas, com espaçamento de 0.4cm:
      Nome do autor e contato (centralizado)
      Disclaimer e número da página (centralizado)
    """
    from config import AUTHOR_NAME, AUTHOR_CONTACT, AUTHOR_DISCLAIMER, AUTHOR_EMAIL, AUTHOR_GITHUB

    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(HexColor("#888888"))

    page_width = A4[0]
    y = 1.0 * cm  # distância do fundo da página

    # Esquerda: autor
    canvas.drawCentredString(
    page_width / 2, y + 0.4 * cm,
    f"{AUTHOR_NAME}  ·  {AUTHOR_CONTACT} ·  {AUTHOR_EMAIL}",
    )

    # Centro: disclaimer
    canvas.drawCentredString(
    page_width / 2, y,
    f"Página {doc.page}  ·  {AUTHOR_DISCLAIMER}",
    )

    # Linha fina cinza acima do rodapé (opcional, fica elegante)
    canvas.setStrokeColor(HexColor("#dddddd"))
    canvas.setLineWidth(0.3)
    canvas.line(2 * cm, y + 0.5 * cm, page_width - 2 * cm, y + 0.5 * cm)

    canvas.restoreState()

def to_pdf(
    questions: list[Question],
    answers: dict[str, int] | None = None,
    title: str = "Simulado CAMA",
    include_answers: bool = True,
) -> bytes:
    """Gera PDF do simulado e retorna os bytes."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=title, author="Simulador CAMA",
    )
    s = _styles()
    story = []

    # cabeçalho
    story.append(Paragraph(_escape(title), s["title"]))
    story.append(Paragraph(
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')} · "
        f"{len(questions)} questões",
        s["meta"],
    ))

    if answers is not None:
        correct = sum(1 for q in questions if answers.get(q.id) == q.answer_idx)
        pct = correct / len(questions) * 100 if questions else 0
        story.append(Paragraph(
            f"<b>Resultado:</b> {correct}/{len(questions)} ({pct:.1f}%)",
            s["meta"],
        ))

    story.append(Spacer(1, 0.4 * cm))

    # questões
    for i, q in enumerate(questions, 1):
        block: list[Flowable] = [
            Paragraph(f"Questão {i}", s["h2"]),
            Paragraph(_escape(q.topic), s["topic"]),
            Paragraph(_escape(q.stem), s["stem"]),
        ]
        for j, opt in enumerate(q.options):
            letter = chr(65 + j)
            text = f"<b>{letter})</b> {_escape(opt)}"
            style = s["option"]
            if include_answers and j == q.answer_idx:
                text = f"✓ <b>{letter})</b> {_escape(opt)}"
                style = s["option_correct"]
            elif answers is not None:
                ua = answers.get(q.id)
                if j == ua and ua != q.answer_idx:
                    text = f"✗ <b>{letter})</b> {_escape(opt)}"
                    style = s["option_wrong"]
            block.append(Paragraph(text, style))

        if include_answers:
            block.append(Paragraph(
                f"Gabarito: {chr(65 + q.answer_idx)}", s["answer"]
            ))
            block.append(Paragraph(
                f"<b>Justificativa:</b> {_escape(q.explanation)}",
                s["explanation"],
            ))
            if q.sources:
                block.append(Paragraph(
                    f"Fontes: {_escape(', '.join(q.sources))}", s["sources"]
                ))

        # KeepTogether tenta manter a questão inteira na mesma página
        story.append(KeepTogether(block))
        story.append(Spacer(1, 0.3 * cm))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buf.getvalue()