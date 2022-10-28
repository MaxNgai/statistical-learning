package data;

import lombok.Data;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

/**
 * @author Max Ngai
 * @since 2022/10/10
 */
@Data
public class Auto {

    private Array2DRowRealMatrix auto;

    public Auto() {
        auto = DataReader.read("Auto");
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
