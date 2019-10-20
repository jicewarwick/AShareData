import os
import sys

sys.path.insert(0, os.path.abspath('../../AShareData'))

# -- Project information -----------------------------------------------------
project = 'AShareData'
copyright = '2019, Ce Ji'
author = 'Ce Ji'
release = '0.0.1'
# -- General configuration ---------------------------------------------------
extensions = ['IPython.sphinxext.ipython_directive',
              'IPython.sphinxext.ipython_console_highlighting',
              # 'sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.autosummary'
              # 'numpydoc'
              ]
templates_path = ['_templates']
exclude_patterns = []
# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# If false, no module index is generated.
html_use_modindex = True
# If false, no index is generated.
html_use_index = True

# -- Options for LaTeX output --------------------------------------------------
latex_paper_size = 'a4'
# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'sampledoc.tex', u'sampledoc Documentation',
     u'John Hunter, Fernando Perez, Michael Droettboom', 'manual'),
]
# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# Additional stuff for the LaTeX preamble.
# latex_preamble = ''

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_use_modindex = True
