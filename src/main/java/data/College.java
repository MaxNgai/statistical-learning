package data;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import java.util.Objects;
import java.util.stream.IntStream;

/**
 * @author Max Ngai
 * @since 2022/12/5
 */
public class College {
    private Array2DRowRealMatrix data;

    public College() {
        data = DataReader.read("College", new CollegeParser());
    }

    public double[][] getX() {
        int[] columns = IntStream.range(0, 18).filter(e -> e != 2 && e != 0).toArray();
        int[] rows = IntStream.range(0, data.getRowDimension()).toArray();
        return data.getSubMatrix(rows, columns).getData();
    }

    public double[] getY() {
        return data.getColumn(2);
    }

    private static class CollegeParser extends DefaultParser {
        @Override
        public double parse(String raw, int columnNo) {
            if (columnNo == 0) {
                return -1D;
            } else if (columnNo == 1) {
                return Objects.equals(raw, "Yes") ? 1 : 0;
            }

            return super.parse(raw, columnNo);
        }
    }
}
