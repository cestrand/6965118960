import weka.classifiers.bayes.BayesNet;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ProjektStart {
	public static void main(String[] args) throws Exception {
		Instances dane_grzyby = DataSource.read("zasoby/mushrooms.arff");
		// wypisanie atrybut�w
		for (Attribute attr : java.util.Collections.list(dane_grzyby.enumerateAttributes())) {
            System.out.println(attr);
        }
		Attribute opisywana = dane_grzyby.attribute("class"); // class to nazwa kolumny m�wi�cej o tym, czy grzyb jest truj�cy (to b�dziemy sprawdza�)
		System.out.println("Ustawiam atrybut klasowy na: " + opisywana);
		dane_grzyby.setClass(opisywana);
		BayesNet bn = new BayesNet();
		bn.buildClassifier(dane_grzyby);
		System.out.println(bn);
		int N = dane_grzyby.numInstances();
		int A = dane_grzyby.numAttributes();
		System.out.println("Ile instancji: " + N);
		System.out.println("Ile atrbut�w: " + A);
	}
}