from funciones import argumentos, load_dataset, data_treatment, setup_ngrok, mlflow_tracking

def main():
    print("Ejecutamos el main")
    args_values = argumentos()
    df = load_dataset()
    X_train, X_test, y_train, y_test = data_treatment(df)
    public_url = setup_ngrok(6000)
    mlflow_tracking(args_values.nombre_job, X_train, X_test, y_train, y_test, 6000)
    input("Presiona Enter para finalizar la sesión de ngrok...")
    ngrok.disconnect(public_url)

if __name__ == "__main__":
    main()
