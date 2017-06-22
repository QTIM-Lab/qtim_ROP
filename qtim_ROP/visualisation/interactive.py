from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
import numpy as np
import pandas as pd
import seaborn as sns


def interactive_tsne(out_file):

    # Create output HTML file
    output_file(out_file)

    cmap = sns.color_palette('colorblind')

    data_dict = dict(
        x=[1, 2, 3, 4, 5],
        y=[2, 5, 8, 2, 7],
        y_true = [0, 1, 2, 1, 0],
        desc=['some image', 'another image', 'woah an image', 'yet another image', 'bored now'],
        imgs = [
            'http://bokeh.pydata.org/static/snake.jpg',
            'http://bokeh.pydata.org/static/snake2.png',
            'http://bokeh.pydata.org/static/snake3D.png',
            'http://bokeh.pydata.org/static/snake4_TheRevenge.png',
            'http://bokeh.pydata.org/static/snakebite.jpg'
        ],
        fonts=['<i>italics</i>',
               '<pre>pre</pre>',
               '<b>bold</b>',
               '<small>small</small>',
               '<del>del</del>'
               ]
    )

    n = 1
    data_dict['colors'] = [
        "#%02x%02x%02x" % (int(cmap[i+n][0] * 255.), int(cmap[i+n][1] * 255.), int(cmap[i+n][2] * 255.)) for i in data_dict['y_true']
    ]

    source = ColumnDataSource(pd.DataFrame(data_dict))

    hover = HoverTool(
            tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="256" alt="@imgs" width="256"
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 17px; font-weight: bold;">@desc</span>
                    <span style="font-size: 15px; color: #966;">[$index]</span>
                </div>
                <div>
                    <span>@fonts{safe}</span>
                </div>
                <div>
                    <span style="font-size: 15px;">Location</span>
                    <span style="font-size: 10px; color: #696;">($x, $y)</span>
                </div>
            </div>
            """
        )

    p = figure(plot_width=800, plot_height=600, tools=[hover],
               title="t-SNE visualization of 'No', 'Pre-Plus' and 'Plus'")

    p.scatter('x', 'y', size=20, source=source, fill_color='colors', fill_alpha=0.7, line_color=None)
    show(p)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-o', '--out', dest='out_file', help="Output HTML file", required=True)
    args = parser.parse_args()

    interactive_tsne(args.out_file)
