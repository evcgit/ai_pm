import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_dir):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = os.path.join(output_dir, f"page{page_num + 1}_img{img_index + 1}.{image_ext}")
            
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
                
            print(f"Saved image {image_filename}")

# extract_images_from_pdf("sample_pdfs/SiteSurvey.pdf", "output_images/")
# python manage.py extract_images detection/sample_pdf/SiteSurvey.pdf detection/output/


