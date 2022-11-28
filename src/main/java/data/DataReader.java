package data;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.MathUtils;
import tool.Macro;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.function.Function;

/**
 * @author Max Ngai
 * @since 2022/9/20
 */
public class DataReader {

    private static final String root = "./src/main/resources/ALL CSV FILES - 2nd Edition/";
    private static final String COMMA = ",";

    public static Array2DRowRealMatrix read(String name) {
        return read(name, new DefaultParser());
    }

    public static Array2DRowRealMatrix read(String name, DefaultParser parser) {
        List<String> raw = readFile(name);
        int n = raw.size() - 1;
        int p = splitByComma(raw.get(0)).size();
        List<double[]> validRows = new ArrayList<>();

        for (int i = 1; i < raw.size(); i++) {
            // start from 1, remove header
            List<String> columns = splitByComma(raw.get(i));
            double[] row = new double[p];
            try {
                for (int j = 0; j < columns.size(); j++) {
                    row[j] = parser.parse(columns.get(j), j);
                }
            } catch (NumberFormatException e) {
                continue; // do nothing, discard the row
            }
            validRows.add(row);
        }

        return Macro.vstack(validRows);
    }

    private static List<String> readFile(String name) {
        Scanner input = null;
        try {
            input = new Scanner(new File(root + name + ".csv"));

            List<String> res = new ArrayList<String>();
            while (input.hasNext()) {
                String raw = input.nextLine();
                res.add(raw.replace("\"", ""));

            }

            return res;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    private static List<String> splitByComma(String str) {
        return Arrays.asList(str.split(COMMA));
    }



}
