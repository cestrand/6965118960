package weka;

public class Matrix {

    static public void show(int[][] M) {
        int m = M.length;
        for (int i = 0; i < m; i++) { // i-ty wiersz
            for (int j = 0; j < m; j++)
                System.out.printf("%3d ", M[i][j]);
            System.out.println();
        }
    }

    static public int showA(int[][] M) {
        int m = M.length;
        int bledy = 0;
        for (int i = 0; i < m; i++) { // i-ty wiersz
            for (int j = 0; j < m; j++) {
                System.out.printf("%3d ", M[i][j]);
                if (i != j) {
                    bledy = bledy + M[i][j];
                }
            }
            System.out.println();
        }
        System.out.println("Liczba bledow: " + bledy);
        return bledy;
    }

    /**
     * Wierszem jest rzeczywista wartość, kolumna to predykcja.
     * @param org
     * @param dec
     * @param numClasses
     * @return
     */
    public static int[][] confMatrix(double[] org, double[] dec, int numClasses) {
        int[][] M = new int[numClasses][numClasses];
        for (int i = 0; i < org.length; i++) {
            M[(int)org[i]][(int)dec[i]]++; // macierz bledow
        }
        return M;
    }
}
