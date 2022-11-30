package tool.modelselection;

import lombok.Data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 *
 *  possible combination of column selected
 *
 * @author Max Ngai
 * @since 2022/11/30
 */
@Data
public class PredictorCombo {
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
    public static Map<Integer, List<PredictorCombo>> getAllCombinations(List<Integer> options, Integer maxPredictors) {
        List<PredictorCombo> rawRes = new ArrayList<PredictorCombo>(){{ add(new PredictorCombo()); }};

        for (Integer option : options) {
            rawRes = rawRes.stream().flatMap(e -> e.fission(option).stream()).collect(Collectors.toList());
        }

        Map<Integer, List<PredictorCombo>> res = rawRes.stream().collect(Collectors.groupingBy(e -> e.getSelected().size()));
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
    private List<PredictorCombo> fission(Integer e) {
        ArrayList<Integer> within = new ArrayList<>(selected);
        within.add(e);
        ArrayList<Integer> without = new ArrayList<>(selected);
        return Arrays.asList(new PredictorCombo(within), new PredictorCombo(without));
    }

    private PredictorCombo(List<Integer> i) { selected = i; }

    private PredictorCombo() {}

    public int getSize() {
        return selected.size();
    }

    public int[] getArray() {
        return selected.stream().mapToInt(e -> e).toArray();
    }

    @Override
    public String toString() {
        return selected.toString();
    }
}
