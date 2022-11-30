package tool.modelselection;

import lombok.Data;
import lombok.ToString;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import tool.CrossValidation;
import tool.model.Model;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/29
 */
public class BestSubsetSelection {

    private RealMatrix X;
    private RealVector Y;
    private int n;
    private int p;
    private Model model;
    private Integer maxPredictors;

    private int[] rowsIndex;
    private List<Score> res;
    private final int kFold = 10;

    /**
     *
     * @param X
     * @param Y
     * @param model
     * @param maxPredictors max amount of predictors that model involved, if null then model might have p predictors at most
     */
    public BestSubsetSelection(double[][] X, double[] Y, Model model, Integer maxPredictors) {
        this.X = new Array2DRowRealMatrix(X);
        this.Y = new ArrayRealVector(Y);
        this.model = model;
        n = X.length;
        p = X[0].length;
        rowsIndex = IntStream.range(0, n).toArray();
        this.maxPredictors = maxPredictors == null ? null : Math.min(p, maxPredictors);
        train();
    }

    private void train() {
        List<Integer> options = IntStream.range(0, p).boxed().collect(Collectors.toList());
        Map<Integer, List<PredictorCombo>> combos = PredictorCombo.getAllCombinations(options, maxPredictors);
        List<Score> ordered = combos.entrySet().stream().parallel()
                .map(i -> {
                    // get combo with min rss
                    List<PredictorCombo> collect = i.getValue().stream()
                            .parallel()
                            .sorted(Comparator.comparing(this::getTrainRss))
                            .collect(Collectors.toList());
                    return collect.get(0);
                })
                .map(c -> {
                    Score score = new Score();
                    score.setK(c.getSize());
                    score.setSelectedX(c);
                    RealMatrix subX = getXByPredictors(c);
                    score.setRss(CrossValidation.kFoldCv(subX, Y, model, kFold));
                    return score;
                })
                .sorted(Comparator.comparingDouble(Score::getRss))
                .collect(Collectors.toList());

        res = ordered;
    }

    private double getTrainRss(PredictorCombo combo) {
        RealMatrix subMatrix = getXByPredictors(combo);
        return model.train(subMatrix, Y).trainRss();
    }

    private RealMatrix getXByPredictors(PredictorCombo combo) {
        int[] columns = combo.getArray();
        RealMatrix subMatrix = X.getSubMatrix(rowsIndex, columns);
        return subMatrix;
    }


    public List<Score> getRes() {
        return res;
    }





}
