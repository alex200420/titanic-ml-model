import argparse
from .titanic_cli import TitanicCLI  # assuming the class TitanicCLI is in titanic_cli.py

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Create the parser for the "train" command
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--train_data', help='path to the training data')
    parser_train.add_argument('--model_path', help='path to save the trained model', default='./output/model/titanic_model.pkl')
    parser_train.add_argument('--preprocessor_path', help='path to save the data preprocessor', default='./output/model/titanic_preprocessor.pkl')
    parser_train.set_defaults(func=lambda args: TitanicCLI(args.model_path, args.preprocessor_path).train(args.train_data))

    # Create the parser for the "predict" command
    parser_predict = subparsers.add_parser('predict')
    parser_predict.add_argument('--predict_data', help='path to the data for prediction')
    parser_predict.add_argument('--out_path', help='path to the data for prediction')
    parser_predict.add_argument('--model_path', help='path to the trained model', default='./output/model/titanic_model.pkl')
    parser_predict.add_argument('--preprocessor_path', help='path to the data preprocessor', default='./output/model/titanic_preprocessor.pkl')
    parser_predict.set_defaults(func=lambda args: TitanicCLI(args.model_path, args.preprocessor_path).predict(args.predict_data, args.out_path))

    # Parse the arguments and call the default function
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()