package kiv.bp.bp_android;

import java.util.concurrent.Callable;


import kiv.bp.bp_android.engines.NeuralEngine;

/**
 * The type Neuro result.
 */
class NeuroResult implements Callable<float[]> {
    private final NeuralEngine engine;

    public NeuroResult(NeuralEngine engine) {
        this.engine = engine;
    }

    @Override
    public float[] call() {
        // Calling the `predict()` method of the `NeuralEngine` class.
        return engine.predict();
    }
}
