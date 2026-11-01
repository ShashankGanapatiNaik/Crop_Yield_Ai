"""PDF report for a stored prediction (ReportLab)."""
from io import BytesIO

from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

GREEN, LIGHT = colors.HexColor("#1f6b45"), colors.HexColor("#e8f3ec")
SEV = {"critical": "#c53030", "warning": "#c05621", "info": "#2b6cb0", "success": "#2f855a"}


def build_pdf(pred, user_name: str) -> bytes:
    r = pred.result
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Yield report #{pred.id}", author="Crop Yield AI")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=GREEN, fontSize=20, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=GREEN, spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)
    el = [Paragraph("Crop Yield Prediction Report", h1),
          Paragraph(f"Report #{pred.id} | {pred.created_at:%d %b %Y %H:%M} UTC | Prepared for {user_name} | Model v{pred.model_version}", small), Spacer(1, 8)]

    eco = r["economics"]
    summary = [["Crop", r["crop"], "Area", f"{r['inputs']['area_ha']:.2f} ha"],
               ["Predicted yield", f"{r['predicted_yield_t_ha']:.2f} t/ha", "80% interval", f"{r['lower_t_ha']:.2f} - {r['upper_t_ha']:.2f} t/ha"],
               ["Total production", f"{r['total_production_t']:.1f} t", "Est. profit", f"INR {eco['profit']:,.0f}"],
               ["Revenue", f"INR {eco['revenue']:,.0f}", "Cost", f"INR {eco['cost']:,.0f}"]]
    t = Table(summary, colWidths=[35 * mm, 45 * mm, 32 * mm, 52 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 9.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.white), ("PADDING", (0, 0), (-1, -1), 6)]))
    el += [t, Paragraph("What drove this prediction (SHAP)", h2),
           Paragraph(f"Baseline average yield {r['base_value_t_ha']:.2f} t/ha. Bars show each factor's contribution in t/ha (green raises, red lowers yield).", body)]

    f = r["top_factors"][::-1]
    d = Drawing(170 * mm, 55 * mm)
    ch = HorizontalBarChart()
    ch.x, ch.y, ch.width, ch.height = 55 * mm, 5 * mm, 105 * mm, 45 * mm
    ch.data = [[x["shap"] for x in f]]
    ch.categoryAxis.categoryNames = [x["label"] for x in f]
    ch.categoryAxis.labels.fontSize = 8
    ch.valueAxis.labels.fontSize = 7
    ch.bars[0].fillColor = GREEN
    for i, x in enumerate(f):
        ch.bars[(0, i)].fillColor = GREEN if x["shap"] >= 0 else colors.HexColor("#c53030")
    d.add(ch)
    el.append(d)

    el.append(Paragraph("Recommendations", h2))
    for rc in r["recommendations"]:
        gain = f" <i>(model estimate: +{rc['expected_gain_t_ha']:.2f} t/ha)</i>" if rc.get("expected_gain_t_ha") else ""
        el.append(Paragraph(f"<font color='{SEV[rc['severity']]}'><b>[{rc['severity'].upper()}] {rc['title']}</b></font><br/>{rc['detail']}{gain}", body))
        el.append(Spacer(1, 4))

    el.append(Paragraph("Input conditions", h2))
    inp = r["inputs"]
    rows = [[k.replace("_", " ").title(), str(v)] for k, v in inp.items()]
    half = (len(rows) + 1) // 2
    grid = [rows[i] + (rows[i + half] if i + half < len(rows) else ["", ""]) for i in range(half)]
    gt = Table(grid, colWidths=[38 * mm, 45 * mm, 38 * mm, 45 * mm])
    gt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("TEXTCOLOR", (0, 0), (0, -1), colors.grey), ("TEXTCOLOR", (2, 0), (2, -1), colors.grey)]))
    el += [gt, Spacer(1, 10),
           Paragraph("Disclaimer: predictions are statistical estimates from a machine-learning model and indicative prices; they are decision support, not a guarantee. "
                     "Confirm fertiliser and lime doses with a local soil test and your agricultural extension officer.", small)]
    doc.build(el)
    return buf.getvalue()
