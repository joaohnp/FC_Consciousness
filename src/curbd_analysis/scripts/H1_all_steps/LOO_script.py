import os
import pickle

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut


def sleep_loo(model, loo, X, Y):
    predictions = []
    true_labels = []
    X_wake = X[Y == 0]
    X_sleep = X[Y == 1]
    Y_wake = Y[Y == 0]
    Y_sleep = Y[Y == 1]
    for train_index, test_index in loo.split(Y_sleep):
        X_train, X_test = (
            np.concatenate((X_wake, X_sleep[train_index])),
            X_sleep[test_index],
        )
        y_train, y_test = (
            np.concatenate((Y_wake, Y_sleep[train_index])),
            Y_sleep[test_index],
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(y_pred[0])
        true_labels.append(y_test[0])
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


def sleep_loo_perm(model, loo, X, Y_perm, Y_true):
    predictions = []
    true_labels = []
    X_wake = X[Y_perm == 0]
    X_sleep = X[Y_perm == 1]
    X_sleep_true = X[Y_true == 1]
    Y_wake = Y_perm[Y_perm == 0]
    Y_sleep = Y_perm[Y_perm == 1]
    Y_sleep_true = Y_true[Y_true == 1]
    for train_index, test_index in loo.split(Y_sleep):
        X_train, X_test = (
            np.concatenate((X_wake, X_sleep[train_index])),
            X_sleep_true[test_index],
        )
        y_train, y_test = (
            np.concatenate((Y_wake, Y_sleep[train_index])),
            Y_sleep_true[test_index],
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(y_pred[0])
        true_labels.append(y_test[0])
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


projectPath = os.path.dirname(os.path.abspath(__name__))
parentPath = os.path.dirname(projectPath)
scoresPath = os.path.join(parentPath, "SleepScores")
curbdPath = os.path.join(parentPath, "Curbd_to_LOO", "LOO-rdy")
qt_val = [0.75, 0.85, 0.95]  # Defining which percentage
for chosen_curbd in os.listdir(curbdPath):
    file_path = os.path.join(curbdPath, chosen_curbd)
    if not os.path.isfile(file_path):
        continue
    print(chosen_curbd)

    # Load curbd data
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    # Load scores
    scores_file_path = os.path.join(scoresPath, chosen_curbd + ".pickle")
    with open(scores_file_path, "rb") as file_scores:
        scores = pickle.load(file_scores)

    # Check scores format
    if len(scores) == 3:
        scores = scores[2]

    for qtd in qt_val:
        curbd = []
        target = []
        strength = []
        for i, app in enumerate(data):
            if np.array(app).size == 1:
                continue
            # if scores[i] == 1 or scores[i] == 2:  # WAKE
            if scores[i] == 4:  # REM or scores[i] == 2:  # WAKE
                #### RE-RUN NREM VS REM #####
                target.append(0)
            elif scores[i] == 3:  # NREM
                target.append(1)
            else:
                continue
            app = np.array(
                [
                    app[0, 1],
                    app[0, 2],
                    app[0, 3],
                    app[1, 0],
                    app[1, 2],
                    app[1, 3],
                    app[2, 0],
                    app[2, 1],
                    app[2, 3],
                    app[3, 0],
                    app[3, 1],
                    app[3, 2],
                ]
            )
            curbd.append(app)
            strength.append(np.mean(abs(app)))

        # Convert lists to numpy arrays
        curbd = np.array(curbd)
        target = np.array(target)
        strength = np.array(strength)
        th = np.nanquantile(strength[target == 0], qtd)
        # print(target)
        # print(strength)
        # print(len(target))
        # print(th)
        app = [
            (
                (target[i] == 1 and strength[i] > th and not np.isnan(strength[i]))
                or (target[i] == 0 and not np.isnan(strength[i]))
            )
            for i in range(len(target))
        ]
        curbd_sel = curbd[app]
        target_sel = target[app]
        strength_sel = strength[app]
        model = svm.SVC(class_weight="balanced")
        loo = LeaveOneOut()
        accu_true = sleep_loo(model, loo, curbd_sel, target_sel)
        print(f"True accuracy: {accu_true}")  # print(accu_true)

        N = 1000
        qt_delta = 0.1
        accu_perm = list()
        for i in range(N):
            target_perm = target_sel
            th_low = np.arange(qtd, 1, qt_delta)
            if th_low[-1] == 1:
                th_high = th_low[1:]
                th_low = th_low[:-1]
            else:
                th_high = th_low[1:]
                th_high = np.append(th_high, 1)
            th_low_values = np.quantile(strength_sel, th_low)
            th_high_values = np.quantile(strength_sel, th_high)
            for j in range(len(th_low)):
                app = np.where(
                    np.logical_and(
                        strength_sel > th_low_values[j],
                        strength_sel < th_high_values[j],
                    )
                )
                app_target = target_sel[app]
                app_target = np.random.permutation(app_target)
                target_perm[app] = app_target
            # accu_perm=cross_validate(model, curbd_sel, target_perm, cv=5)
            accu_perm.append(
                sleep_loo_perm(model, loo, curbd_sel, target_perm, target_sel)
            )

        # Calculate statistics
        accu_perm = np.array(accu_perm)
        # print(f"accu_perm: {accu_perm}")
        p_value = len(accu_perm[accu_perm > accu_true]) / len(accu_perm)
        print(f"p_value: {p_value}")
        # Save results
        dict_curbd = {
            "accu_true": accu_true,
            "accu_perm": accu_perm,
            "p_value": p_value,  # Added p_value to output
        }
        output_filename = (
            chosen_curbd + "_REMvsNREM" + str(qtd) + ".pickle"
        )  # Added better filename
        print("saving " + output_filename)
        with open(output_filename, "wb") as f:
            pickle.dump(dict_curbd, f)
