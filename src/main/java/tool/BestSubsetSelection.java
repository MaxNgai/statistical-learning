package tool;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
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
    private List<Combo> bestPredictorSet;
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
        Map<Integer, List<Combo>> combos = Combo.getAllCombinations(options, maxPredictors);
        List<Combo> ordered = combos.entrySet().stream().parallel()
                .map(i -> {
                    // get combo with min rss
                    List<Combo> collect = i.getValue().stream()
                            .parallel()
                            .sorted(Comparator.comparing(this::getTrainRss))
                            .collect(Collectors.toList());
                    return collect.get(0);
                })
                .sorted(Comparator.comparing(combo -> CrossValidation.kFoldCv(X, Y, model, kFold)))
                .collect(Collectors.toList());
        bestPredictorSet = ordered;
    }

    private double getTrainRss(Combo combo) {
        int[] columns = combo.getSelected().stream().mapToInt(e -> e).toArray();
        RealMatrix subMatrix = X.getSubMatrix(rowsIndex, columns);
        return model.train(subMatrix, Y).trainRss();
    }

    public List<Combo> getBestPredictorSet() {
        return bestPredictorSet;
    }

    /**
     * performance score of a model
     */
    @Data
    private static class Score {
        /**
         * number of selected predictors
         */
        private int k;

        /**
         * column index of selected predictors,
         */
        private Combo selectedX;


        private double rss;
        private double rSquare;
    }

    /**
     * possible combination of column selected
     */
    @Data
    private static class Combo {
        /**
         * index of selected columns
         */
        private List<Integer> selected = new ArrayList<>();

        /**
         * given an array of options(int), generate all the combinations of these options.
         * If there are K options, then it will generate 2^k - 1 combinations.
         *
         * in order to manage them better, it returns a map grouping by how many options are selected
         * @param options
         * @return eg. if K = 3, then returns map like
         * {
         *     1:[[1],[2],[3]],
         *     2:[[1,2],[2,3],[3,1]],
         *     3:[1,2,3]
         * }
         *
         */
        public static Map<Integer, List<Combo>> getAllCombinations(List<Integer> options, Integer maxPredictors) {
            List<Combo> rawRes = new ArrayList<Combo>(){{ add(new Combo()); }};

            for (Integer option : options) {
                rawRes = rawRes.stream().flatMap(e -> e.fission(option).stream()).collect(Collectors.toList());
            }

            Map<Integer, List<Combo>> res = rawRes.stream().collect(Collectors.groupingBy(e -> e.getSelected().size()));
            res.remove(0);

            if (maxPredictors != null) {
                for (int i = maxPredictors + 1; i <= options.size(); i++) {
                    res.remove(i);
                }
            }

            return res;
        }

        /**
         * like nuclear fission, when consider one property,
         * there will be two status of it, selected or not selected in a new combo
         * @param e
         * @return
         */
        private List<Combo> fission(Integer e) {
            ArrayList<Integer> within = new ArrayList<>(selected);
            within.add(e);
            ArrayList<Integer> without = new ArrayList<>(selected);
            return Arrays.asList(new Combo(within), new Combo(without));
        }

        private Combo(List<Integer> i) { selected = i; }

        private Combo() {}

        public int getSize() {
            return selected.size();
        }
    }
}
