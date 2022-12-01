package tool.modelselection;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.Pair;
import tool.CrossValidation;
import tool.model.Model;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/30
 */
public abstract class ModelSelection {
    RealMatrix X;
    RealVector Y;
    int n;
    int p;
    Model model;
    Integer maxPredictors;

    int[] rowsIndex;
    final int kFold = 10;

    /**
     * best models with K predictors and their score
     */
    List<Score> cacheScoreTrainedByAllData;

    ModelSelection(double[][] X, double[] Y, Model model, Integer maxPredictors) {
        this.X = new Array2DRowRealMatrix(X);
        this.Y = new ArrayRealVector(Y);
        this.model = model;
        n = X.length;
        p = X[0].length;
        rowsIndex = IntStream.range(0, n).toArray();
        this.maxPredictors = maxPredictors == null ? p : Math.min(p, maxPredictors);
    }

    public List<Score> getCacheScoreTrainedByAllData() {
        return cacheScoreTrainedByAllData;
    }

    /**
     * this method do the same as textbook, which a R language function
     * @param x
     */
    protected abstract List<Score> regSubset(RealMatrix x, RealVector y);

    public int chooseKWithCv() {
        List<CrossValidation.DataSet> dataSets = CrossValidation.kFoldDataSplit(X, Y, kFold);

        // kFold * p matrix
        Map<Integer, List<Score>> kAndScore = dataSets.parallelStream()
                .flatMap(d -> {
                    // size of p, each element is the best model of its K size model(k predictors)
                    return regSubset(d.getTrainX(), d.getTrainY()).parallelStream()
                            .map(modelK -> {
                                int rows = d.getTestX().getRowDimension();
                                RealMatrix subMatrix = d.getTestX().getSubMatrix(IntStream.range(0, rows).toArray(), modelK.getSelectedX().getArray());
                                double testRss = modelK.getModel().testRss(subMatrix, d.getTestY());
                                modelK.setTestRss(testRss);
                                return modelK;
                            });
                }).collect(Collectors.groupingBy(e -> e.getSelectedX().getSize()));

        List<Pair<Integer, Double>> kAndRss = kAndScore.entrySet().stream().map(e ->
                new Pair<Integer, Double>(e.getKey(), e.getValue().stream().mapToDouble(Score::getTestRss).average().getAsDouble()))
                .sorted(Comparator.comparingInt(Pair::getFirst))
                .collect(Collectors.toList());

        System.out.println("kAndRss = " + kAndRss);
        return kAndRss.stream().sorted(Comparator.comparing(e -> e.getSecond())).findFirst().get().getFirst();
    }
}
