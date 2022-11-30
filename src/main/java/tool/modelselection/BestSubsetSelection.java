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
public class BestSubsetSelection extends ModelSelection {

    /**
     *
     * @param X
     * @param Y
     * @param model
     * @param maxPredictors max amount of predictors that model involved, if null then model might have p predictors at most
     */
    public BestSubsetSelection(double[][] X, double[] Y, Model model, Integer maxPredictors) {
        super(X, Y, model, maxPredictors);
        train();
    }

    private void train() {
        List<Integer> options = IntStream.range(0, p).boxed().collect(Collectors.toList());
        Map<Integer, List<PredictorCombo>> combos = PredictorCombo.getAllCombinations(options, maxPredictors);
        List<Score> ordered = combos.entrySet().stream().parallel()
                .map(i -> {
                    // get combo with min rss
                    return i.getValue().stream()
                            .parallel()
                            .sorted(Comparator.comparing(this::getTrainRss))
                            .findFirst()
                            .get();
                })
                .map(c -> {
                    Score score = new Score();
                    score.setK(c.getSize());
                    score.setSelectedX(c);
                    RealMatrix subX = getXByPredictors(c);
                    score.setRss(CrossValidation.kFoldCv(subX, Y, model, kFold) * n);
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



}
