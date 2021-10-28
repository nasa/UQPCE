from datetime import datetime
import os
import subprocess
from subprocess import PIPE

from warnings import warn, showwarning

try:
    from mpi4py.MPI import DOUBLE as MPI_DOUBLE, COMM_WORLD as MPI_COMM_WORLD
except:
    warn('Ensure that all required packages are installed.')
    exit()

import PCE_Codes._helpers as _help

showwarning = _help._warn

comm = MPI_COMM_WORLD
size = comm.size
rank = comm.rank
is_manager = (rank == 0)


class Report():
    """
    Base Report class.
    """

    def __init__(self):
        self.body = ''  # initialize empty string to append

    def _write(self, report_name, body, report_file):
        """
        Inputs: report_name- the title of the report
                body- the body of the text to write
                report_file- the output file for the text to be written to
        
        Writes the report LaTeX file.
        """

        beg_of_file = str(
                '\\documentclass[12pt, letterpaper, twoside]{article}\n'
                '\\usepackage[utf8]{inputenc}\n\\title{'
                f'{report_name}' '}\n\\usepackage{graphicx}\n\setlength{'
                '\\parindent}{0pt}\n\\date{'
                f'{datetime.today().strftime("%Y-%m-%d")}'
                '}\n\n\\begin{document}\n'
                '\\begin{titlepage}\n\\maketitle\n\\end{titlepage}\n\n'
        )

        if is_manager:
            with open(report_file, 'w') as rep:
                rep.write(beg_of_file)
                rep.write(body)
                rep.write('\n\\end{document}')

    def add_stat(self, name, value, thresh_value):
        """
        Inputs: name- the name of the statistic (ex. 'R squared')
                value- the value of the statistic
                thresh_value- the minimum/maximum value that the statistic 
                should have
        
        Adds the statistic 'name' to the string that will be written to the 
        file.
        """
        if is_manager:
            stat = (f'{name} has value {value} but should be closer '
                    f'to {thresh_value}.\n\n')

            self.body = ''.join((self.body, stat))

    def add_text(self, text):
        """
        Inputs: text- the text to be added to the .tex file
        
        Adds 'text' to the string that will be written to the file.
        """
        self.body = ''.join((self.body, text, '\n\n'))

    def add_plot(self, figure_path, width=8, caption=''):
        """
        Inputs: figure_path- the path(s) from the current working directory to 
                the figure that the user wants to include in the report
                width- the width of the figure in the report (default: 8cm)
                caption- the caption to write with the figure; if no caption is 
                given, the figure_path will be written as the caption
        
        Adds a figure for the given figure path to the string that will be 
        written to the file.
        """
        if is_manager:

            if isinstance(figure_path, list):
                fig_count = len(figure_path)
            else:
                fig_count = 1
                figure_path = [figure_path]

            for idx in range(fig_count):

                if caption == '':
                    curr_caption = figure_path[idx].replace('_', '\_')
                else:
                    curr_caption = caption

                new_image = (
                    '\\begin{figure}[hbt!]\n\\centering\n\\includegraphics[width='
                    f'{width}' 'cm]{' f'{figure_path[idx]}' '}\n\\caption{'
                    f'{curr_caption}' '}\n\\end{figure}\n\n'
                )

                self.body = ''.join((self.body, new_image))

    def add_pagebreak(self):
        """
        Adds a pagebreak to the LaTeX file.
        """
        if is_manager:
            self.body = ''.join((self.body, '\\pagebreak\n\n'))

    def add_section(self, title):
        """
        Adds a new Section to the LaTeX file.
        """
        if is_manager:
            sect = ('\section*{' f'{title}' '}\n\n')
            self.body = ''.join((self.body, sect))

    def clear(self):
        """
        Clears all of the text to be written to the file.
        """
        self.body = ''


class UQPCEReport(Report):
    """
    Report class that writes all of the stats that appear less than ideal  to 
    a file.
    """

    def __init__(self):
        super().__init__()
        self.report_name = 'UQPCE Report'

    def write(self, directory=None):
        """
        Inputs: directory- the directory in which to place the files (optional)
        
        Writes the UQPCE report LaTeX file.
        """
        report_file = 'uqpce_report.tex'

        if is_manager:
            if directory is None:
                directory = os.getcwd()

            path = os.path.join(directory, report_file)

            self._write(self.report_name, self.body, path)
