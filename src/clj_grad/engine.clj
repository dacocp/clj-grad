(ns clj-grad.engine
  (:require [taoensso.timbre :refer [spy]]))

;; TODO: implement and assoc the `:backward` function for all operations
;; TODO: build function `backward` which performs backpropagation on the
;; computation graph rooted at the given Value

;; Define a Value record to store scalar values and their gradients
(defrecord Value [data grad children op label backward]
  Object
  (toString [this]
    #_(str "Value(data=" data ", grad=" grad ")")
    label))

(defn value
  "Create a Value with the given data, children, and operation."
  ([data] (value data [] nil "const" nil))
  ([data label] (value data [] nil label nil))
  ([data children op label backward]
   (->Value data 0M (set children) op label backward)))

(defn- add-op [v1 v2]
  "Addition operation for Value instances."
  (let [out (value (+ (:data v1) (:data v2)))]
    (assoc out :children [v1 v2]
               :op "+"
               :label (str (:label v1) " + " (:label v2))
               )))

(defn- mul-op [v1 v2]
  "Multiplication operation for Value instances."
  (let [out (value (* (:data v1) (:data v2)))]
    (assoc out :children [v1 v2]
               :op "*"
               :label (str (:label v1) " * " (:label v2)))))

(defn- pow-op [v p]
  "Power operation for Value instances."
  (let [out (value (Math/pow (:data v) p))]
    (assoc out :children [v]
               :op (str "**" p)
               :label (str (:label v) "**" p))))

(defn- tanh-op [v]
  "Apply the hyperbolic tangent function to a Value instance."
  (let [out (value (Math/tanh (:data v)))]
    (assoc out :children [v]
               :op "tanh"
               :label (str "tanh(" (:label v) ")"))))

(defn- relu-op [v]
  "ReLU operation for Value instances."
  (let [out (value (if (neg? (:data v)) 0 (:data v)))]
    (assoc out :children [v]
               :op "ReLU"
               :label (str "ReLU(" (:label v) ")"))))

(defn backward [v]
  "Perform backpropagation on the computation graph rooted at the given Value."
  ;; TODO
  nil)

(defn add
  ([v1 v2]
   "Add two Value instances or a Value instance and a number."
   (if (and (instance? Value v1) (instance? Value v2))
     (add-op v1 v2)
     (if (instance? Value v1)
       (add-op v1 (value v2))
       (add-op (value v1) v2))))
  ([v1 v2 & more]
   "Add multiple Value instances or Value instances and numbers."
   (reduce add (add v1 v2) more)))

(defn mul [v1 v2]
  "Multiply two Value instances or a Value instance and a number."
  (if (and (instance? Value v1) (instance? Value v2))
    (mul-op v1 v2)
    (if (instance? Value v1)
      (mul-op v1 (value v2))
      (mul-op (value v1) v2))))

(defn pow [v p]
  "Raise a Value instance to the power of a number."
  (pow-op v p))

(defn tanh [v]
  "Apply the hyperbolic tangent function to a Value instance."
  (tanh-op v))

(defn relu [v]
  "Apply the ReLU function to a Value instance."
  (if (instance? Value v)
    (relu-op v)
    (relu-op (value v))))

(defn neg [v]
  "Negate a Value instance."
  (mul v -1))

(defn sub [v1 v2]
  "Subtract two Value instances or a Value instance and a number."
  (add v1 (neg v2)))

(defn div [v1 v2]
  "Divide two Value instances or a Value instance and a number."
  (mul v1 (pow v2 -1)))
