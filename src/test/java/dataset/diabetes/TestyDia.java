package dataset.diabetes;
import java.util.Random;

import weka.Matrix;
import weka.SiecB;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestyDia{

	public static void main(String[] args) throws Exception {
		Instances dane = DataSource.read("zasoby/weka-data/diabetes.arff");
		dane.setClassIndex(dane.numAttributes() - 1);

		SiecB bn = SiecB.createHillClimbGlobal(2);
		bn.buildClassifier(dane);
		bn.PrintScore();
		bn.PrintData();

		System.out.println("xWalidacja OneOut");
		int[][] M = bn.confMatrixOfXVal(dane, 0);
		Matrix.show(M);

		
		/*
		 * ......zadanie: wyznaczyc HISTOGRAM (10K prob lub wiecej) dla kazdej z 4
		 * wartosci wystepujacych w tablicy M; ocenic podobienstwo do rozkladu
		 * normalnego, w tym srednia i wariancje (odchylenie standardowe).
		 */
		int q = 2;
		System.out.println("xWalidacja " + q + "-krotna");
		int err, Err = 0, delta = 0;
		
		for (int i = 1; i < 10000; i++) { // sprawdzic dla skrajnych: 246, 340, 581, 661
			// petla szuka maksymalnych wartosci
			M = bn.confMatrixOfXVal(dane, q, new Random(i));
			err = M[0][1] + M[1][0];
			if (Err < err) {
				Err = err;
				System.out.println(i + ": Err = " + Err);
			}
			if (delta < -M[0][1] + M[1][0]) {
				delta = -M[0][1] + M[1][0];
				System.out.println(i + ": delta = " + delta);
			}
		}
		M = bn.confMatrixOfXVal(dane, 2, new Random(95924)); // skrajny przyp.?
		Matrix.show(M);
		
	}
}
//Tablica maksymalnej czestosci - dla 50K prob (q=2)!:
//423  77
// 89 179  

//xWalidacja OneOut
//418  82 
// 87 181 

//246: Err = 184
//340: delta = 34
//581: Err = 186
//661: delta = 42
//12071: Err = 186
//12252: Err = 187
//13185: delta = 40
//14410: delta = 41
//20013: Err = 187
//20025: delta = 34
//20282: delta = 37
//20289: delta = 40
//20409: delta = 41
//31456: Err = 188
//32417: delta = 40
//35578: delta = 41
//36254: Err = 190
//37442: delta = 42
//45606: delta = 43
//49131: Err = 189
//52242: Err = 192
//65000: delta = 45
// xWalidacja 2-krotna
// 440 60
// 105 163
//67443: Err = 194
// xWalidacja 2-krotna
// 408 92
// 102 166
//95924: Err = 195
// xWalidacja 2-krotna
// 403 97
// 98 170
