package tool.modelselection;

import lombok.Data;
import lombok.ToString;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import tool.CrossValidation;
import tool.Macro;
import tool.model.Model;

import java.util.*;
import java.util.function.Function;
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
        cacheScoreTrainedByAllData = regSubset(this.X, this.Y);
    }

    protected List<Score> regSubset(RealMatrix input, RealVector output) {
        List<Integer> options = IntStream.range(0, p).boxed().collect(Collectors.toList());
        int[] selectedRows = IntStream.range(0, input.getRowDimension()).toArray();
        Map<Integer, List<PredictorCombo>> combos = PredictorCombo.getAllCombinations(options, maxPredictors);
        return combos.entrySet().stream().parallel()
                .map(i -> {

                    // get combo with min rss
                    return i.getValue().stream()
                            .parallel()
                            .map(combo -> {
                                int[] columns = combo.getArray();
                                RealMatrix subMatrix = input.getSubMatrix(selectedRows, columns);
                                Model train = model.train(subMatrix, output);
                                return new Score(combo, train);
                            })
                            .sorted(Comparator.comparingDouble(Score::getTrainRss))
                            .findFirst()
                            .get();
                })
                .collect(Collectors.toList());

    }




}
