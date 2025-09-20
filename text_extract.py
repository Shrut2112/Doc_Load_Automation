import pymupdf
from PIL import Image,ImageOps,ImageFilter
import io
import pytesseract



doc = pymupdf.open("finance.pdf")

for page in doc:
    text = page.get_text("blocks")
    
    if text:
        for t in text:
            print(t[4])
            
        
    
    images = page.get_images()
    
    if images:
        for img_index,img in enumerate(images):
            xref = img[0] # unique image id
            act_imag = doc.extract_image(xref)

            image_bytes = act_imag["image"]

            image = Image.open(io.BytesIO(image_bytes))
            
            results = pytesseract.image_to_osd(image, config='--psm 0', output_type=pytesseract.Output.DICT)
            angle = results["rotate"]

            image = image.rotate(angle,expand=True)
            image.save("img.png")
            filtered_image = image.filter(ImageFilter.MedianFilter(size=3))
            gray_img = ImageOps.grayscale(filtered_image)

            scale = 300 / 72
            base_w = min(int(gray_img.width * scale), 2500)
            base_h = min(int(gray_img.height * scale), 2500)
            gray_img_re = gray_img.resize((base_w, base_h), Image.LANCZOS)

                    
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            n = len(data['text'])
            lines = {}
            for i in range(n):
                if int(data['conf'][i]) > 0 and data['text'][i].strip():
                    y = data['top'][i]
                    x = data['left'][i]
                    text = data['text'][i]
                    line_key = round(y / 10)   # Tune the bucket size as needed
                    if line_key not in lines:
                        lines[line_key] = []
                    lines[line_key].append((x, text))
                    
            # Now combine words within each line and then lines in order
            ordered_lines = []
            for line in sorted(lines.keys()):
                words = [t[1] for t in sorted(lines[line], key=lambda z: z[0])]
                ordered_lines.append(" ".join(words))

            ordered_text = " ".join(ordered_lines)
            print(ordered_text)
    break
            

        
        
