import pptx

prs = pptx.Presentation(r'c:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\template_poster.pptx')
slide = prs.slides[0]
for i, s in enumerate(slide.shapes):
    txt = s.text_frame.text.replace('\n', ' ')[:60] if s.has_text_frame else '(no text)'
    print(f'{i:02d}: left={s.left/914400:5.1f} top={s.top/914400:5.1f} w={s.width/914400:5.1f} h={s.height/914400:5.1f} name={s.name:<22} text={txt}')
