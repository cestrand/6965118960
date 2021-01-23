package weka;

import weka.core.Instances;

public class Podzial {
    public Instances l;
    public Instances r;

    public Podzial(Instances l, Instances r) {
        this.l = l;
        this.r = r;
    }
}
