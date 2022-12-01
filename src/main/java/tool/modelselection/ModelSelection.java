package tool.modelselection;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import tool.CrossValidation;
import tool.model.Model;

import java.util.Comparator;
import java.util.List;
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
        List<List<Score>> matrix = dataSets.parallelStream()
                .map(d -> {
                    // size of p, each element is the best model of its K size model(k predictors)
                    return regSubset(d.getTrainX(), d.getTrainY()).parallelStream()
                            .map(modelK -> {
                                double testRss = modelK.getModel().testRss(d.getTestX(), d.getTestY());
                                modelK.setTestRss(testRss);
                                return modelK;
                            })
                            .sorted(Comparator.comparingInt(e -> e.getSelectedX().getSize()))
                            .collect(Collectors.toList());
                }).collect(Collectors.toList());

        ArrayRealVector sumRssOfKFold = matrix.parallelStream()
                .map(fold -> fold.parallelStream().mapToDouble(Score::getTestRss).toArray())
                .map(ArrayRealVector::new)
                .reduce(ArrayRealVector::add).get();

        RealVector realVector = sumRssOfKFold.mapDivide(matrix.size()); // mean rss of k-fold on different size of best model
        System.out.println(realVector);
        return realVector.getMinIndex();
    }
}
