import os

import imageio
import openpyxl

def load():
    """Yields examples from this dataset."""
    data_path = os.path.join(os.path.dirname(__file__), "data")
    wkbk_path = os.path.join(data_path, "counts.xlsx")
    wkbk = openpyxl.load_workbook(filename=wkbk_path)
    for example_name in wkbk.sheetnames:
        wksht = wkbk[example_name]
        example_path = os.path.join(data_path, example_name)
        image = imageio.imread(os.path.join(example_path, "plate.jpg"))
        dcol = wksht['A22'].value - wksht['A20'].value
        dxdcol = (wksht['C22'].value - wksht['C20'].value) / dcol
        dydcol = (wksht['D22'].value - wksht['D20'].value) / dcol
        drow = wksht['B21'].value - wksht['B20'].value
        dxdrow = (wksht['C21'].value - wksht['C20'].value) / drow
        dydrow = (wksht['D21'].value - wksht['D20'].value) / drow
        for col in range(12):
            for row in range(8):
                count = wksht[chr(ord('A') + col) + str(row + 1)].value

                xmin = round(wksht['C20'].value + dxdcol*(col-0.5) +
                             dxdrow*(row-0.5))
                xmin = max(xmin, 0)
                xmax = xmin + round(dxdcol + dxdrow)
                xmax = min(xmax, image.shape[1])

                ymin = round(wksht['D20'].value + dydcol*(col-0.5) +
                             dydrow*(row-0.5))
                ymin = max(ymin, 0)
                ymax = ymin + round(dydcol + dydrow)
                ymax = min(ymax, image.shape[0])

                patch = image[ymin:ymax, xmin:xmax, :]
                yield (patch, count)
