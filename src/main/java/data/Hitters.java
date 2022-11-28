package data;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/11/28
 */
public class Hitters {
    private Array2DRowRealMatrix data;

    public Hitters() {
        Array2DRowRealMatrix credit = DataReader.read("Hitters", new HittersDataParser());
        data = credit;
    }


    public double[] getY() {
        return data.getColumn(18);
    }

    public double[][] getX() {
        return data.getSubMatrix(
                IntStream.range(0, data.getRowDimension()).toArray(),
                IntStream.range(0, data.getColumnDimension()).filter(e -> e != 18).toArray()
        ).getData();
    }

    private static class HittersDataParser extends DefaultParser {
        @Override
        public double parse(String raw, int columnNo) {
            if (Objects.equals(raw, "A")) {
                return 1D;
            }

            if (Objects.equals(raw, "N")) {
                return 0D;
            }

            if (Objects.equals(raw, "E")) {
                return 1D;
            }

            if (Objects.equals(raw, "W")) {
                return 0D;
            }

            if (Objects.equals(raw, "NA")) {
              throw new NumberFormatException("invalidValue, discard");
            }

            return super.parse(raw, columnNo);
        }
    }
}
