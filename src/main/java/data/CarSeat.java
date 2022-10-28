package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * @author Max Ngai
 * @since 2022/10/11
 */
@Data
public class CarSeat {
    private Array2DRowRealMatrix data;

    public CarSeat() {
        data = DataReader.read("Carseats", new CarSeatParser());

    }

    public double[] getSales() {
        return data.getColumn(0);
    }

    public double[] getPrice() {
        return data.getColumn(5);
    }

    public double[] getUrban() {
        return data.getColumn(9);
    }

    public double[] getUS() {
        return data.getColumn(10);
    }


    static class CarSeatParser extends DefaultParser {
        @Override
        public double parse(String raw, int columns) {
            switch (raw) {
                case "Yes" : return 1D;
                case "No" : return -1D;
                case "Bad" : return 1D;
                case "Medium" : return 2D;
                case "Good" : return 3D;
                default: return super.parse(raw, columns);
            }

        }
    }

}
