import gradio as gr
from craft_hw_ocr import OCR

ocr = OCR.load_models()

def do_ocr(inp):
  img, results = OCR.detection(inp, ocr[2])
  bboxes, text = OCR.recoginition(img, results, ocr[0], ocr[1])
  return OCR.visualize(img, results), text
  
inputs = gr.components.Image()
o1 = gr.components.Image()
o2 = gr.components.Textbox()

title = "CRAFT TrOCR"
description = "OCR of both handwriting and printed text using CRAFT Text detector and TrOCR recognition, detection of lines and extraction of them are happening here because TrOCR pre-trained models are modelled on IAM lines dataset and the same needs to be implemented here."
examples=[['example_1.png'],['example_2.jpg']]

article = "<p style='text-align: center'><a href='https://github.com/Vishnunkumar/craft_hw_ocr' target='_blank'>craft_hw_ocr</a></p><p style='text-align: center'> <p style='text-align: center'><a href='https://github.com/fcakyon/craft-text-detector' target='_blank'>craft-text-detector</a></p><p style='text-align: center'>"
gr.Interface(fn=do_ocr, inputs=inputs, outputs=[o1, o2], title=title, description=description, article=article, examples=examples).launch()
