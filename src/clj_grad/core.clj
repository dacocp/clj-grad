(ns clj-grad.core
  (:require [clj-grad.engine :as e]
            [clj-grad.nn :refer [mlp]]))

;; For now, this ns is only used for testing `Value` record and
;; the operations defined for it, and the MLP construction from
;; `nn` ns
(comment
  ; inputs x1,x2
  (def x1 (e/value 2.0 "x1"))
  (clojure.pprint/pprint x1)
  (def x2 (e/value 0.0 "x2"))
  (clojure.pprint/pprint x2)

  ; weights w1,w2
  (def w1 (e/value -3.0 "w1"))
  (clojure.pprint/pprint w1)
  (def w2 (e/value 1.0 "w2"))
  (clojure.pprint/pprint w2)

  ; bias of the neuron
  (def b (e/value 6.8813735870195432 "b"))
  (clojure.pprint/pprint b)

  ; x1*w1 + x2*w2 + b
  (def x1w1 (e/mul x1 w1))
  (clojure.pprint/pprint x1w1)

  (def x2w2 (e/mul x2 w2))
  (clojure.pprint/pprint x2w2)

  (def x1w1x2w2 (e/add x1w1 x2w2))
  (clojure.pprint/pprint x1w1x2w2)
  (def n (e/add x1w1x2w2 b))
  (clojure.pprint/pprint n)
  (def o (e/tanh n))
  (clojure.pprint/pprint o)
  )

(comment
  ;; Test the constructor of the MLP

  ;; input vector x
  (def x (map e/value [1.0 2.0 3.0] ["x1" "x2" "x3"]))

  ;; MLP
  (def net (mlp 3 [2 1]))
  net
  (net x)

  ;; invoke (forward-propagate)
  (def fwd (net x))
  fwd
  ;(def act (net x))
  ;(clojure.pprint/pprint act)
)
