try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import sudoku

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    text = sudoku.solve(filename)  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text  # Then we will print the text in the image
