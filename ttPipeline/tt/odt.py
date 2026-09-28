"""
OdtDocument — just enough of OpenDocument text for a results report: headings,
paragraphs, tables, figures with captions.

A thin layer over odfpy so the report code says what it means ("a table of
these rows", "this figure, this wide") rather than assembling XML nodes.
"""

import os

from PIL import Image
from odf.draw import Frame, Image as DrawImage
from odf.opendocument import OpenDocumentText
from odf.style import (GraphicProperties, ParagraphProperties, Style,
                       TableCellProperties, TableColumnProperties,
                       TextProperties)
from odf.table import Table, TableCell, TableColumn, TableRow
from odf.text import H, P, Span


class OdtDocument(object):

    def __init__(self, title):
        self.document = OpenDocumentText()
        self.styles = {}
        self._figureCount = 0
        self._tableCount = 0
        self._defineStyles()
        self.paragraph(title, "Title")

    # ------------------------------------------------------------------ #

    def _defineStyles(self):
        automatic, named = self.document.automaticstyles, self.document.styles

        def paragraphStyle(name, size, bold=False, italic=False,
                           above="0.1cm", below="0.2cm", align=None,
                           family="paragraph"):
            style = Style(name=name, family=family)
            text = {"fontsize": size}
            if bold:
                text["fontweight"] = "bold"
            if italic:
                text["fontstyle"] = "italic"
            style.addElement(TextProperties(**text))
            paragraph = {"margintop": above, "marginbottom": below}
            if align:
                paragraph["textalign"] = align
            style.addElement(ParagraphProperties(**paragraph))
            named.addElement(style)
            self.styles[name] = style

        paragraphStyle("Title", "20pt", bold=True, below="0.5cm")
        paragraphStyle("Heading1", "15pt", bold=True, above="0.5cm")
        paragraphStyle("Heading2", "12.5pt", bold=True, above="0.35cm")
        paragraphStyle("Body", "10.5pt", align="justify")
        paragraphStyle("Caption", "9pt", italic=True, below="0.4cm")
        paragraphStyle("Code", "9pt", above="0cm", below="0cm")
        paragraphStyle("TableText", "9pt", above="0.02cm", below="0.02cm")

        bold = Style(name="Bold", family="text")
        bold.addElement(TextProperties(fontweight="bold"))
        automatic.addElement(bold)
        self.styles["Bold"] = bold

        cell = Style(name="Cell", family="table-cell")
        cell.addElement(TableCellProperties(border="0.5pt solid #999999",
                                            padding="0.06cm"))
        automatic.addElement(cell)
        self.styles["Cell"] = cell

        header = Style(name="HeaderCell", family="table-cell")
        header.addElement(TableCellProperties(border="0.5pt solid #999999",
                                              padding="0.06cm",
                                              backgroundcolor="#e8eef4"))
        automatic.addElement(header)
        self.styles["HeaderCell"] = header

        frame = Style(name="FigureFrame", family="graphic")
        frame.addElement(GraphicProperties(anchortype="paragraph",
                                           horizontalpos="center",
                                           horizontalrel="paragraph",
                                           wrap="none"))
        automatic.addElement(frame)
        self.styles["FigureFrame"] = frame

    # ------------------------------------------------------------------ #

    def nextTable(self):
        """The number the next table will get, for referring to it first."""
        return self._tableCount + 1

    def nextFigure(self):
        return self._figureCount + 1

    def heading(self, text, level=1):
        self.document.text.addElement(
            H(outlinelevel=level, stylename=self.styles["Heading%d" % level],
              text=text))

    def paragraph(self, text, style="Body"):
        """
        A paragraph. **double asterisks** mark bold runs, which is the only
        inline formatting the report needs.
        """
        paragraph = P(stylename=self.styles[style])
        for index, piece in enumerate(text.split("**")):
            if not piece:
                continue
            if index % 2:
                paragraph.addElement(Span(stylename=self.styles["Bold"],
                                          text=piece))
            else:
                paragraph.addText(piece)
        self.document.text.addElement(paragraph)

    def code(self, text):
        for line in text.strip("\n").splitlines():
            self.document.text.addElement(P(stylename=self.styles["Code"],
                                            text=line))

    def table(self, header, rows, caption=None, widthsCm=None):
        self._tableCount += 1
        name = "Table%d" % self._tableCount
        table = Table(name=name)
        for position in range(len(header)):
            column = Style(name="%sCol%d" % (name, position),
                           family="table-column")
            width = (widthsCm[position] if widthsCm else 16.0 / len(header))
            column.addElement(TableColumnProperties(columnwidth="%.2fcm"
                                                    % width))
            self.document.automaticstyles.addElement(column)
            table.addElement(TableColumn(stylename=column))
        table.addElement(self._row(header, "HeaderCell", bold=True))
        for row in rows:
            table.addElement(self._row(row, "Cell"))
        self.document.text.addElement(table)
        if caption:
            self.paragraph("Table %d. %s" % (self._tableCount, caption),
                           "Caption")

    def _row(self, values, cellStyle, bold=False):
        row = TableRow()
        for value in values:
            cell = TableCell(stylename=self.styles[cellStyle])
            paragraph = P(stylename=self.styles["TableText"])
            if bold:
                paragraph.addElement(Span(stylename=self.styles["Bold"],
                                          text=str(value)))
            else:
                paragraph.addText(str(value))
            cell.addElement(paragraph)
            row.addElement(cell)
        return row

    def figure(self, path, caption, widthCm=15.5):
        """An image scaled to `widthCm`, keeping its aspect ratio."""
        self._figureCount += 1
        with Image.open(path) as image:
            width, height = image.size
        heightCm = widthCm * height / float(width)
        stored = self.document.addPicture(path)
        frame = Frame(stylename=self.styles["FigureFrame"],
                      width="%.2fcm" % widthCm, height="%.2fcm" % heightCm,
                      anchortype="paragraph",
                      name="Figure%d" % self._figureCount)
        frame.addElement(DrawImage(href=stored))
        holder = P(stylename=self.styles["Body"])
        holder.addElement(frame)
        self.document.text.addElement(holder)
        self.paragraph("Figure %d. %s" % (self._figureCount, caption),
                       "Caption")

    def save(self, path):
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.document.save(path)
        return path
