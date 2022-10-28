package graph;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.Scatter3DTrace;
import tech.tablesaw.plotly.traces.ScatterTrace;

/**
 * simple tool for display scatter plot in browser
 *
 * @author Max Ngai
 * @since 2022/10/10
 */
public class ScatterPlot {

    public static void see(double[] x, double[] y) {
        see("x", x, "y", y);
    }

    public static void see(String xName, double[] x, String yName, double[] y) {
        ScatterTrace build = ScatterTrace.builder(DoubleColumn.create(xName, DoubleArrayList.wrap(x)),
                DoubleColumn.create(yName, DoubleArrayList.wrap(y))).build();
        System.out.println(build.asJavascript(1));
        Plot.show(Figure.builder().addTraces(build)
                .layout(Layout.builder().height(900).width(900).build())
                .build());
    }

    public static void see(String xName, double[] x,
                           String yName, double[] y,
                           String zName, double[] z) {
        Scatter3DTrace build = Scatter3DTrace.builder(DoubleColumn.create(xName, DoubleArrayList.wrap(x)),
                DoubleColumn.create(yName, DoubleArrayList.wrap(y)),
                DoubleColumn.create(zName, DoubleArrayList.wrap(z))

                ).build();
        Plot.show(Figure.builder().addTraces(build)
                .addTraces()
                .layout(Layout.builder().height(900).width(1400).build()).build());
    }

    public static void see(double[] x,
                           double[] y,
                           double[] z) {
        see("x", x, "y", y, "z", z);
    }

}
