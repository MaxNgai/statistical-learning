package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/10/10
 */
@Data
public class Auto {

    private Array2DRowRealMatrix auto;

    private List<Integer> illegalRows = Arrays.asList(32,126,330,336,354);

    public Auto() {
        auto = DataReader.read("Auto");
        int[] ints = IntStream.range(0, 397).filter(e -> {
            // remove rows that has horse power of '?'
            return !illegalRows.contains(e);
        }).toArray();
        auto = (Array2DRowRealMatrix)auto.getSubMatrix(ints, new int[]{0,1,2,3,4,5,6,7,8});
    }

    public double[] getMpg() {
        return auto.getColumn(0);
    }

    public double[] getCylinder() {
        return auto.getColumn(1);

    }

    public double[] getDisplacement() {
        return auto.getColumn(2);

    }

    public double[] getHorsePower() {
        return auto.getColumn(3);
    }

    public double[] getWeight() {
        return auto.getColumn(4);
    }

    public double[] getAccelerate() {
        return auto.getColumn(5);
    }

    public double[] getYear() {
        return auto.getColumn(6);
    }

    public double[] getOrigin() {
        return auto.getColumn(7);
    }


}
