(ns clj-grad.nn
  (:require [clj-grad.engine :as e]))

;; Define a protocol for neural network components
(defprotocol Module
  (zero-grad [this])
  (parameters [this]))

;; Define Neuron type
(defrecord Neuron [w b nonlin]
  Module
  (zero-grad [this]
    ;; TODO
    nil)
  (parameters [_]
    (concat w [b]))
  Object
  (toString [_]
    (str (if nonlin "ReLU" "Linear") "Neuron(" (count w) ")"))
  clojure.lang.IFn
  (invoke [this x]
    (let [wx (zipmap w x)
          lin-comb (reduce (fn [acc [w x]]
                             (e/add acc (e/mul w x)))
                           (apply e/mul (first wx))
                           (rest wx))
          act (e/add b lin-comb)]
      (if nonlin (e/relu act) act))))

(defn build-neuron
  "Create a neuron with the given number of input weights and non-linearity."
  [nin nonlin & {:keys [layer_no]}]

  (->Neuron
    (map (fn [idx]
           (e/value (- (rand 2) 1.0M) (str "w" layer_no (inc idx))))
         (range nin))
    (e/value (- (rand 2) 1.0M) (str "b" layer_no))
    nonlin))

;; Define Layer type
(defrecord Layer [neurons]
  Module
  (zero-grad [_]
    (doseq [n neurons]
      (zero-grad n)))
  (parameters [_]
    (mapcat parameters neurons))
  Object
  (toString [_]
    (str "Layer of [" (clojure.string/join ", " (map str neurons)) "]"))
  clojure.lang.IFn
  (invoke [this x]
    (map (fn [neuron] (neuron x)) neurons)))

(defn build-layer
  "Create a layer with the given number of input and output neurons."
  [nin nout nonlin & {:keys [layer_no]}]
  (println "Building layer" layer_no "with" nin "inputs and" nout "outputs")
  (->Layer (repeatedly nout #(build-neuron nin nonlin :layer_no layer_no))))

;; Define MLP type
(defrecord MLP [layers]
  Module
  (zero-grad [_]
    (doseq [l layers]
      (zero-grad l)))
  (parameters [_]
    (mapcat parameters layers))
  Object
  (toString [_]
    (str "MLP of [" (clojure.string/join ", " (map str layers)) "]"))
  clojure.lang.IFn
  (invoke [this x]
    (reduce (fn [input layer] (layer input)) x layers))
  (applyTo [this args]
    (.invoke this (first args))))

(defn mlp
  "Create a multi-layer perceptron with the given input size and a sequence of output sizes."
  [nin nouts]
  (let [sz (cons nin nouts)
        layer-count (count nouts)]
    (->MLP (for [i (range layer-count)]
             (build-layer (nth sz i) (nth sz (inc i)) (not= i (dec layer-count)) :layer_no (inc i))))))
