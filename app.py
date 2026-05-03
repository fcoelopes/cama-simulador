"""Simulador CAMA — Streamlit app."""
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import DATA_DIR
from ingest import ingest_all, collection_stats, reset_collection
from llm import PROVIDERS, build_client
from quiz import (
    generate_questions, load_cached, count_cached, list_topics,
    record_attempts, inspect_retrieval, build_prompt_preview,
    generate_batch,
)
from export import to_markdown, to_pdf
import dashboard as dash


st.set_page_config(page_title="Simulador CAMA", page_icon="📚", layout="wide")


# ---------- estado ----------

def init_state():
    defaults = {
        "questions": [],       # questões da sessão atual
        "answers": {},         # {q_id: idx_escolhido}
        "submitted": False,
        "current_idx": 0,
        "session_id": None,    # gerado ao iniciar simulado
        "tracked": False,      # evita gravar 2x se a página renderizar de novo
        "last_provider": None, # para detectar troca e resetar API key
        "last_model": None,
        "api_key": "",         # gerenciada em session_state pra poder limpar
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------- sidebar: configuração ----------

with st.sidebar:
    st.title("⚙️ Configuração")

    st.subheader("LLM")
    provider = st.selectbox("Provider", list(PROVIDERS.keys()), index=0)
    model = st.selectbox("Modelo", PROVIDERS[provider]["models"])

    # Limpa API key se trocou provider ou modelo
    if (st.session_state.last_provider is not None
            and (st.session_state.last_provider != provider
                 or st.session_state.last_model != model)):
        st.session_state.api_key = ""
        st.caption("🔑 API key foi limpa por causa da troca de modelo/provider.")

    st.session_state.last_provider = provider
    st.session_state.last_model = model

    api_key = st.text_input(
        "API Key", type="password",
        value=st.session_state.api_key,
        key="api_key_input",
        help="Não é persistida em disco — fica só na sessão. "
             "É limpa automaticamente ao trocar de provider/modelo.",
    )
    st.session_state.api_key = api_key

    st.divider()
    st.subheader("Base de conhecimento")
    stats = collection_stats()
    st.metric("Chunks indexados", stats["chunks"])
    st.metric("Arquivos-fonte", stats["sources"])

    if stats["source_list"]:
        with st.expander("Arquivos indexados"):
            for s in stats["source_list"]:
                st.caption(f"• {s}")

    st.caption(f"📂 Coloque arquivos em: `{DATA_DIR}`")
    st.caption("Suportados: PDF, DOCX, MD, TXT")

    # Upload via UI
    uploaded = st.file_uploader(
        "📤 Adicionar arquivo(s)",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        help="Os arquivos vão para a pasta data/. Lembre de clicar em Indexar depois.",
    )
    if uploaded:
        saved, skipped, errors = [], [], []
        for f in uploaded:
            # sanitização básica do nome (sem path traversal)
            safe_name = Path(f.name).name
            target = DATA_DIR / safe_name
            if target.exists() and target.stat().st_size == f.size:
                skipped.append(safe_name)
                continue
            try:
                target.write_bytes(f.getbuffer())
                saved.append(safe_name)
            except Exception as e:
                errors.append(f"{safe_name}: {e}")

        if saved:
            st.success(f"✅ Salvos em data/: {', '.join(saved)}")
        if skipped:
            st.info(f"↩️ Já existiam (mesmo tamanho): {', '.join(skipped)}")
        if errors:
            st.error("Erros: " + "; ".join(errors))

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Indexar", use_container_width=True):
            with st.spinner("Indexando..."):
                progress = st.progress(0)
                status = st.empty()

                def ingest_cb(i, total, name):
                    progress.progress(i / max(total, 1))
                    status.caption(f"[{i}/{total}] {name}")

                result = ingest_all(progress_cb=ingest_cb)
                progress.empty()
                status.empty()
                st.success(
                    f"✅ {result['files']} novos arquivo(s), "
                    f"{result['chunks']} chunk(s). Pulados: {result['skipped']}"
                )
                if result["errors"]:
                    st.warning("Erros: " + "; ".join(result["errors"][:3]))
                st.rerun()

    with col_b:
        if st.button("🗑️ Resetar", use_container_width=True):
            reset_collection()
            st.success("Índice apagado.")
            st.rerun()


# ---------- corpo principal ----------

st.title("📚 Simulador CAMA")
st.caption("Certified Asset Management Assessor — questionários gerados via RAG sobre sua base de conhecimento")

tab_play, tab_gen, tab_dash, tab_debug = st.tabs([
    "🎯 Simulado", "✨ Gerar novas questões", "📊 Dashboard", "🔍 Debug RAG",
])


# ---------- aba: gerar ----------

with tab_gen:
    st.subheader("Gerar questões via LLM")

    if stats["chunks"] == 0:
        st.warning("Nenhum conteúdo indexado ainda. Adicione arquivos em `data/` e clique em **Indexar** na sidebar.")
    elif not api_key:
        st.info("Informe a API key do provider escolhido na sidebar.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input(
                "Tópico / assunto",
                placeholder="Ex: ISO 55001 Liderança e compromisso, RCM, gestão de risco em ativos físicos…",
            )
        with col2:
            n = st.number_input("Qtd. questões", 1, 20, 5)

        mode = st.radio(
            "Modo de geração",
            options=["strict", "lenient"],
            format_func=lambda m: {
                "strict": "🎯 Rigoroso — qualidade > quantidade (pode gerar menos)",
                "lenient": "📚 Amplo — atinge a quantidade pedida com variações do mesmo conceito",
            }[m],
            horizontal=False,
            help=(
                "**Rigoroso**: o modelo retorna apenas questões com forte ancoragem no contexto. "
                "Pode retornar menos do que pedido se o tópico for estreito. Use pra simulado real.\n\n"
                "**Amplo**: o modelo cobre o tópico sob diferentes ângulos pra atingir N. Use pra "
                "drill/revisão de tópicos específicos quando você quer mais volume de prática."
            ),
        )

        if st.button("Gerar", type="primary", disabled=not topic):
            try:
                llm = build_client(provider, api_key, model)
            except Exception as e:
                st.error(f"Erro ao iniciar o LLM: {e}")
            else:
                with st.spinner(f"Gerando até {n} questões com {provider}/{model} ({mode})…"):
                    try:
                        new_qs = generate_questions(llm, topic, n=n, persist=True, mode=mode)
                    except Exception as e:
                        st.error(f"Falha na geração: {e}")
                        new_qs = []

                if new_qs:
                    if len(new_qs) < n:
                        st.warning(
                            f"⚠️ {len(new_qs)} de {n} questões geradas e salvas no cache. "
                            f"O modelo entendeu que o contexto sustenta apenas essa quantidade. "
                            f"Pra mais volume sobre o mesmo tópico, tente o modo **Amplo** ou "
                            f"refine o tópico (ex: divida em sub-tópicos)."
                        )
                    else:
                        st.success(f"✅ {len(new_qs)} questão(ões) gerada(s) e salva(s) no cache.")
                    with st.expander("Pré-visualizar"):
                        for i, q in enumerate(new_qs, 1):
                            st.markdown(f"**{i}. {q.stem}**")
                            for j, opt in enumerate(q.options):
                                marker = "✅" if j == q.answer_idx else "▫️"
                                st.markdown(f"{marker} {chr(65 + j)}) {opt}")
                            st.caption(f"💡 {q.explanation}")
                            st.divider()
                else:
                    st.error(
                        "Não foi possível gerar questões. Possíveis causas: "
                        "(1) tópico não está coberto pelo material — use a aba 🔍 Debug RAG; "
                        "(2) modelo retornou JSON malformado — tente regenerar ou troque pro Gemini Pro/Sonnet."
                    )

        # ---- Geração em lote ----
        st.divider()
        with st.expander("📦 Gerar lote de tópicos (cobertura ampla)"):
            st.caption(
                "Cole uma lista de tópicos (um por linha) e o sistema gera N questões "
                "de cada em sequência. Útil pra cobrir um documento denso "
                "(ex: as 39 subject areas do AM Landscape, ou todas as cláusulas da ISO 55001)."
            )

            preset = st.selectbox(
                "Carregar preset",
                [
                    "(personalizado)",
                    "ISO 55001 — todas as cláusulas",
                    "AM Landscape — 6 áreas principais",
                    "AM Landscape — 39 subject areas (cobertura completa)",
                ],
            )

            preset_text = ""
            if preset == "ISO 55001 — todas as cláusulas":
                preset_text = "\n".join([
                    "ISO 55001 cláusula 4 Contexto da organização",
                    "ISO 55001 cláusula 5.1 Liderança e Compromisso",
                    "ISO 55001 cláusula 5.2 Política de gestão de ativos",
                    "ISO 55001 cláusula 5.3 Papéis, responsabilidades e autoridades",
                    "ISO 55001 cláusula 6.1 Ações para tratar riscos e oportunidades",
                    "ISO 55001 cláusula 6.2 Objetivos de gestão de ativos e SAMP",
                    "ISO 55001 cláusula 7.1 Recursos",
                    "ISO 55001 cláusula 7.2 Competência",
                    "ISO 55001 cláusula 7.3 Conscientização",
                    "ISO 55001 cláusula 7.4 Comunicação",
                    "ISO 55001 cláusula 7.5 Requisitos de informação",
                    "ISO 55001 cláusula 7.6 Informação documentada",
                    "ISO 55001 cláusula 8.1 Planejamento e controle operacional",
                    "ISO 55001 cláusula 8.2 Gestão de mudanças",
                    "ISO 55001 cláusula 8.3 Terceirização",
                    "ISO 55001 cláusula 9.1 Monitoramento, medição, análise e avaliação",
                    "ISO 55001 cláusula 9.2 Auditoria interna",
                    "ISO 55001 cláusula 9.3 Análise crítica pela direção",
                    "ISO 55001 cláusula 10 Não conformidade e ação corretiva",
                ])
            elif preset == "AM Landscape — 6 áreas principais":
                preset_text = "\n".join([
                    "Strategy and Planning no GFMAM AM Landscape",
                    "Asset Management Decision-Making no GFMAM AM Landscape",
                    "Lifecycle Delivery no GFMAM AM Landscape",
                    "Asset Information no GFMAM AM Landscape",
                    "Organisation and People no GFMAM AM Landscape",
                    "Risk and Review no GFMAM AM Landscape",
                ])
            elif preset == "AM Landscape — 39 subject areas (cobertura completa)":
                preset_text = "\n".join([
                    # Strategy & Planning (4)
                    "Asset Management Policy", "Asset Management Strategy and Objectives",
                    "Demand Analysis", "Strategic Planning", "Asset Management Planning",
                    # Decision Making (5)
                    "Capital Investment Decision-Making", "Operations and Maintenance Decision-Making",
                    "Lifecycle Value Realisation", "Resourcing Strategy", "Shutdowns and Outage Strategy",
                    # Lifecycle Delivery (10)
                    "Technical Standards and Legislation", "Asset Creation and Acquisition",
                    "Systems Engineering", "Configuration Management", "Maintenance Delivery",
                    "Reliability Engineering", "Asset Operations", "Resource Management",
                    "Shutdown and Outage Management", "Fault and Incident Response",
                    "Asset Decommissioning and Disposal",
                    # Asset Information (5)
                    "Asset Information Strategy", "Asset Information Standards",
                    "Asset Information Systems", "Data and Information Management",
                    "Procurement and Supply Chain Management",
                    # Organisation & People (4)
                    "Asset Management Leadership", "Organisational Structure",
                    "Organisational Culture", "Competence Management",
                    # Risk & Review (10)
                    "Risk Assessment and Management", "Contingency Planning and Resilience Analysis",
                    "Sustainable Development", "Management of Change",
                    "Assets Performance and Health Monitoring", "Asset Management System Monitoring",
                    "Management Review Audit and Assurance", "Asset Costing and Valuation",
                    "Stakeholder Engagement", "Asset Management System",
                ])

            topics_text = st.text_area(
                "Tópicos (um por linha)",
                value=preset_text,
                height=200,
                placeholder="Tópico 1\nTópico 2\nTópico 3\n...",
            )
            n_per = st.number_input("Questões por tópico", 1, 10, 3)

            topics_list = [t.strip() for t in topics_text.splitlines() if t.strip()]
            if topics_list:
                est_total = len(topics_list) * n_per
                est_calls = len(topics_list)
                st.caption(
                    f"📊 Estimativa: **{len(topics_list)} tópicos × {n_per} questões = "
                    f"até {est_total} questões** ({est_calls} chamadas ao LLM)"
                )

            if st.button("🚀 Gerar lote", type="primary",
                         disabled=not topics_list,
                         key="gen_batch_btn"):
                try:
                    llm = build_client(provider, api_key, model)
                except Exception as e:
                    st.error(f"Erro ao iniciar o LLM: {e}")
                else:
                    progress = st.progress(0)
                    status = st.empty()

                    def batch_cb(idx, total, topic, n_gen):
                        progress.progress(idx / total)
                        status.caption(f"[{idx}/{total}] {topic} → {n_gen} questões")

                    with st.spinner("Gerando lote..."):
                        try:
                            batch = generate_batch(
                                llm, topics_list, n_per_topic=n_per, mode=mode,
                                progress_cb=batch_cb,
                            )
                        except Exception as e:
                            st.error(f"Falha no lote: {e}")
                            batch = None

                    progress.empty()
                    status.empty()

                    if batch:
                        retries_msg = ""
                        if batch.get("retries", 0) > 0:
                            retries_msg = f" ({batch['retries']} retry(s) feito(s))"
                        st.success(
                            f"✅ Lote concluído. **{batch['total_generated']} questões** "
                            f"geradas em {len(topics_list)} tópicos{retries_msg}."
                        )
                        # tabela com resumo
                        summary = [
                            {"Tópico": t, "Questões geradas": len(qs),
                             "Status": "✅" if len(qs) >= n_per else ("⚠️" if len(qs) > 0 else "❌")}
                            for t, qs in batch["by_topic"].items()
                        ]
                        st.dataframe(summary, use_container_width=True, hide_index=True)

                        if batch["errors"]:
                            with st.expander(f"⚠️ {len(batch['errors'])} erro(s) após retries"):
                                for err in batch["errors"]:
                                    st.caption(f"• {err}")


# ---------- aba: simulado ----------

with tab_play:
    cached_total = count_cached()
    topics = list_topics()

    if cached_total == 0:
        st.info("Você ainda não tem questões no cache. Vá na aba **Gerar novas questões**.")
    else:
        st.caption(f"💾 {cached_total} questões em cache · {len(topics)} tópicos")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            topic_filter = st.selectbox(
                "Filtrar por tópico",
                ["(todos)"] + topics,
            )
        with col2:
            n_quiz = st.number_input("Tamanho do simulado", 1, 50, 10)
        with col3:
            st.write("")
            st.write("")
            if st.button("🎯 Iniciar simulado", type="primary", use_container_width=True):
                topic_arg = None if topic_filter == "(todos)" else topic_filter
                st.session_state.questions = load_cached(n=n_quiz, topic=topic_arg)
                st.session_state.answers = {}
                st.session_state.submitted = False
                st.session_state.current_idx = 0
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.tracked = False
                st.rerun()

        st.divider()

        qs = st.session_state.questions
        if qs:
            if not st.session_state.submitted:
                # ---- modo prova ----
                idx = st.session_state.current_idx
                q = qs[idx]

                st.markdown(f"**Questão {idx + 1} de {len(qs)}** · _{q.topic}_")
                st.markdown(f"### {q.stem}")

                choice = st.radio(
                    "Escolha:",
                    options=list(range(len(q.options))),
                    format_func=lambda i: f"{chr(65 + i)}) {q.options[i]}",
                    index=st.session_state.answers.get(q.id),
                    key=f"radio_{q.id}",
                )
                if choice is not None:
                    st.session_state.answers[q.id] = choice

                nav1, nav2, nav3 = st.columns([1, 1, 1])
                with nav1:
                    if st.button("⬅ Anterior", disabled=(idx == 0), use_container_width=True):
                        st.session_state.current_idx -= 1
                        st.rerun()
                with nav2:
                    answered = len(st.session_state.answers)
                    st.caption(f"Respondidas: {answered}/{len(qs)}")
                with nav3:
                    if idx < len(qs) - 1:
                        if st.button("Próxima ➡", use_container_width=True):
                            st.session_state.current_idx += 1
                            st.rerun()
                    else:
                        if st.button("✅ Finalizar", type="primary", use_container_width=True):
                            st.session_state.submitted = True
                            st.rerun()

                # export antes de finalizar (versão "prova em branco")
                with st.expander("📥 Exportar prova (sem gabarito)"):
                    md_blank = to_markdown(qs, answers=None, include_answers=False,
                                            title="Simulado CAMA — Prova")
                    pdf_blank = to_pdf(qs, answers=None, include_answers=False,
                                       title="Simulado CAMA — Prova")
                    ts = datetime.now().strftime("%Y%m%d_%H%M")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "⬇️ PDF",
                            data=pdf_blank, #pdf_blank
                            file_name=f"simulado_{ts}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    with c2:
                       st.download_button(
                           "⬇️ Markdown",
                           data=md_blank,
                           file_name=f"simulado_{ts}.md",
                           mime="text/markdown",
                           use_container_width=True,
                      )

            else:
                # ---- modo resultado ----
                # registra tentativas (uma vez só)
                if not st.session_state.tracked and st.session_state.session_id:
                    record_attempts(
                        st.session_state.session_id,
                        qs,
                        st.session_state.answers,
                    )
                    st.session_state.tracked = True

                correct = sum(
                    1 for q in qs
                    if st.session_state.answers.get(q.id) == q.answer_idx
                )
                pct = correct / len(qs) * 100
                st.metric("Acerto", f"{correct}/{len(qs)}", f"{pct:.1f}%")

                if pct >= 70:
                    st.success("🎉 Bom desempenho — nível de aprovação CAMA.")
                elif pct >= 50:
                    st.warning("⚠️ Desempenho intermediário. Reveja os erros abaixo.")
                else:
                    st.error("🔴 Abaixo do esperado. Foco na revisão dos tópicos.")

                col_md, col_pdf, col_new = st.columns(3)
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                md_full = to_markdown(qs, answers=st.session_state.answers,
                                       include_answers=True,
                                       title="Simulado CAMA — Resultado")
                pdf_full = to_pdf(qs, answers=st.session_state.answers,
                                  include_answers=True,
                                  title="Simulado CAMA — Resultado")
                with col_md:
                    st.download_button(
                        "⬇️ Markdown (gabarito)",
                        data=md_full,
                        file_name=f"resultado_{ts}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with col_pdf:
                    st.download_button(
                        "⬇️ PDF (gabarito)",
                        data=pdf_full,
                        file_name=f"resultado_{ts}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                with col_new:
                    if st.button("🔁 Novo simulado", use_container_width=True):
                        st.session_state.questions = []
                        st.session_state.answers = {}
                        st.session_state.submitted = False
                        st.session_state.current_idx = 0
                        st.session_state.session_id = None
                        st.session_state.tracked = False
                        st.rerun()

                st.divider()

                for i, q in enumerate(qs, 1):
                    user_ans = st.session_state.answers.get(q.id)
                    is_correct = user_ans == q.answer_idx
                    icon = "✅" if is_correct else "❌"
                    with st.expander(f"{icon} Questão {i}: {q.stem[:80]}…"):
                        st.markdown(f"**{q.stem}**")
                        for j, opt in enumerate(q.options):
                            prefix = ""
                            if j == q.answer_idx:
                                prefix = "✅ "
                            elif j == user_ans and not is_correct:
                                prefix = "❌ "
                            st.markdown(f"{prefix}**{chr(65 + j)})** {opt}")
                        st.info(f"💡 **Justificativa:** {q.explanation}")
                        st.caption(f"📄 Fontes: {', '.join(q.sources)}")


# ---------- aba: dashboard ----------

with tab_dash:
    st.subheader("📊 Métricas de desempenho")

    overall = dash.overall_stats()

    if overall.total_attempts == 0:
        st.info("Você ainda não finalizou nenhum simulado. Termine ao menos uma sessão pra ver suas métricas aqui.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessões concluídas", overall.sessions)
        c2.metric("Questões respondidas", overall.total_attempts)
        c3.metric("Taxa de acerto", f"{overall.accuracy * 100:.1f}%")
        c4.metric("Questões únicas vistas", overall.unique_questions)

        st.divider()

        # evolução por sessão
        st.markdown("### 📈 Evolução por sessão")
        sessions = dash.by_session()
        if len(sessions) >= 2:
            chart_data = {
                "Sessão": [f"#{i+1}" for i in range(len(sessions))],
                "Acerto (%)": [s["accuracy"] * 100 for s in sessions],
            }
            st.line_chart(chart_data, x="Sessão", y="Acerto (%)", height=280)
        else:
            st.caption("Faça pelo menos 2 simulados pra ver a evolução temporal.")

        # por tópico
        st.markdown("### 🎯 Desempenho por tópico")
        topics_data = dash.by_topic()
        if topics_data:
            chart = {
                "Tópico": [t["topic"] for t in topics_data],
                "Acerto (%)": [t["accuracy"] * 100 for t in topics_data],
            }
            st.bar_chart(chart, x="Tópico", y="Acerto (%)",
                         horizontal=True, height=max(280, 40 * len(topics_data)))

            with st.expander("Ver tabela detalhada"):
                table = [
                    {
                        "Tópico": t["topic"],
                        "Tentativas": t["total"],
                        "Acertos": t["correct"],
                        "Taxa": f"{t['accuracy'] * 100:.1f}%",
                    }
                    for t in topics_data
                ]
                st.dataframe(table, use_container_width=True, hide_index=True)

            worst = topics_data[0]
            if worst["accuracy"] < 0.7:
                st.warning(
                    f"⚠️ Tópico mais fraco: **{worst['topic']}** "
                    f"({worst['accuracy'] * 100:.1f}% de acerto em {worst['total']} questões). "
                    f"Considere gerar mais questões desse tópico pra reforçar."
                )

        # questões mais difíceis
        st.divider()
        st.markdown("### 🔥 Questões mais difíceis (pra você)")
        hardest = dash.hardest_questions(limit=10)
        if hardest:
            for h in hardest:
                acc_pct = h["accuracy"] * 100
                stem_short = h["stem"][:120] + ("…" if len(h["stem"]) > 120 else "")
                st.markdown(
                    f"- **{acc_pct:.0f}%** de acerto em {h['attempts']} tentativa(s) · "
                    f"_{h['topic']}_<br>{stem_short}",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Sem dados suficientes ainda.")

        st.divider()
        if st.button("🗑️ Apagar histórico de tentativas",
                     help="Remove todas as tentativas registradas. As questões em cache ficam intactas."):
            dash.reset_history()
            st.success("Histórico apagado.")
            st.rerun()


# ---------- aba: debug RAG ----------

with tab_debug:
    st.subheader("🔍 Debug RAG — inspeção do pipeline de geração")
    st.caption(
        "Use esta aba pra investigar quando as questões saem ruins. "
        "Ela mostra exatamente o que o LLM recebe como contexto."
    )

    if stats["chunks"] == 0:
        st.warning("Nenhum conteúdo indexado. Adicione arquivos e indexe primeiro.")
    else:
        debug_topic = st.text_input(
            "Tópico de teste",
            placeholder="Ex: ISO 55001 cláusula 6 — Planejamento e gestão de risco",
            key="debug_topic",
        )
        col_k, col_n = st.columns(2)
        with col_k:
            debug_k = st.slider("Top-K chunks", 1, 15, 6,
                                help="Quantos chunks recuperar. Default = 6.")
        with col_n:
            debug_n = st.slider("N questões (preview)", 1, 10, 5,
                                help="Pra calcular o tamanho do prompt final.")

        if debug_topic:
            st.divider()

            # --- 1. Retrieval ---
            st.markdown("### 1️⃣ Chunks recuperados")
            st.caption(
                "**Distância cosseno**: 0 = idêntico ao tópico, 2 = oposto. "
                "Retrieval bom geralmente fica abaixo de 0.7. "
                "Se o pior chunk passar de 1.0, o material provavelmente "
                "não cobre esse tópico — gere sobre outro assunto."
            )

            try:
                results = inspect_retrieval(debug_topic, k=debug_k)
            except Exception as e:
                st.error(f"Erro no retrieval: {e}")
                results = []

            if not results:
                st.warning("Nenhum chunk recuperado.")
            else:
                # Sumário visual
                avg_dist = sum(r["distance"] for r in results if r["distance"] is not None) / len(results)
                worst = max((r["distance"] for r in results if r["distance"] is not None), default=0)
                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks recuperados", len(results))
                c2.metric("Distância média", f"{avg_dist:.3f}")
                c3.metric("Pior distância", f"{worst:.3f}",
                          delta="ruim" if worst > 1.0 else "ok",
                          delta_color="inverse" if worst > 1.0 else "normal")

                # Diagnóstico automático
                if avg_dist > 0.9:
                    st.error(
                        "🔴 **Retrieval ruim**: distância média alta. "
                        "Possíveis causas: (a) tópico não está coberto pelo material, "
                        "(b) chunks têm muito ruído (OCR sujo), "
                        "(c) tópico muito vago — tente termos mais específicos da norma."
                    )
                elif avg_dist > 0.7:
                    st.warning(
                        "🟡 **Retrieval mediano**: alguns chunks podem não ser muito relevantes. "
                        "Considere reformular o tópico com termos exatos da norma."
                    )
                else:
                    st.success("🟢 **Retrieval bom**: chunks aparentam ser relevantes.")

                # Lista de chunks
                for r in results:
                    dist_color = "🟢" if r["distance"] < 0.7 else ("🟡" if r["distance"] < 1.0 else "🔴")
                    with st.expander(
                        f"{dist_color} **#{r['rank']}** · dist={r['distance']:.3f} · "
                        f"📄 {r['source']} (chunk {r['chunk_idx']})"
                    ):
                        st.code(r["text"], language="text", wrap_lines=True)

            # --- 2. Prompt final ---
            st.divider()
            st.markdown("### 2️⃣ Prompt enviado ao LLM")
            st.caption(
                "Este é exatamente o que o modelo recebe. Se o contexto aqui "
                "não tem a informação que você espera, nenhum prompt vai salvar."
            )

            try:
                preview = build_prompt_preview(debug_topic, n=debug_n, k=debug_k)
            except Exception as e:
                st.error(f"Erro montando prompt: {e}")
                preview = None

            if preview:
                c1, c2 = st.columns(2)
                c1.metric("Tamanho do contexto", f"{preview['context_chars']:,} chars")
                c2.metric("Fontes únicas", len(preview["sources"]))

                with st.expander("📜 System prompt", expanded=False):
                    st.code(preview["system"], language="text", wrap_lines=True)

                with st.expander("📜 User prompt completo", expanded=False):
                    st.code(preview["user"], language="text", wrap_lines=True)

            # --- 3. Sugestões acionáveis ---
            st.divider()
            st.markdown("### 3️⃣ Checklist de diagnóstico")
            st.markdown("""
            Se as questões estão ruins, verifique nesta ordem:

            **a) O retrieval está trazendo o conteúdo certo?**
            Veja a seção 1. Se distância média > 0.9, o problema é busca, não LLM.
            → Reformule o tópico com vocabulário exato da norma (ex: "context of the organization"
            em vez de "contexto organizacional").

            **b) Os chunks estão limpos?**
            Abra um expander acima e leia o texto. Se vir caracteres bagunçados, palavras
            grudadas, ou tabelas viradas em texto incompreensível, o problema é OCR/parsing.
            → Reextraia o PDF ou substitua por uma versão com texto nativo.

            **c) Os chunks cobrem o conceito completo?**
            Se um chunk corta no meio de uma definição importante, aumente CHUNK_SIZE em config.py
            (de 800 pra 1200, por exemplo) e reindexe.

            **d) O modelo é capaz?**
            Modelos pequenos (Flash-Lite, Haiku, GPT-4o-mini) têm mais tendência a alucinar
            em conteúdo técnico denso. Teste com Sonnet ou Gemini 2.5 Pro como controle.

            **e) O system prompt precisa ajuste?**
            Só vá pra cá depois de eliminar a, b, c, d. Adicionar restrições no prompt
            quando o problema é retrieval/chunks só piora os outros casos.
            """)