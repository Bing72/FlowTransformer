#  FlowTransformer 2023 by liamdm / liam@riftcs.com
#  CICFlowMeter 데이터셋 데모 스크립트

import os
import pandas as pd

from framework.dataset_specification import NamedDatasetSpecifications
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from implementations.classification_heads import *
from implementations.input_encodings import *
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer
from implementations.transformers.named_transformers import *

def run_cicflowmeter_dataset(dataset_path, cache_folder=None):
    """
    CICFlowMeter로 생성된 CSV 파일을 FlowTransformer로 학습
    
    Args:
        dataset_path (str): CICFlowMeter로 생성된 CSV 파일 경로
        cache_folder (str): 캐시 폴더 경로 (선택사항)
    """
    
    print(f"CICFlowMeter 데이터셋 로딩: {dataset_path}")
    
    # 다양한 인코딩 옵션
    encodings = [
        NoInputEncoder(),
        RecordLevelEmbed(64),
        CategoricalFeatureEmbed(EmbedLayerType.Dense, 16),
        CategoricalFeatureEmbed(EmbedLayerType.Lookup, 16),
        CategoricalFeatureEmbed(EmbedLayerType.Projection, 16),
        RecordLevelEmbed(64, project=True)
    ]

    # 다양한 분류 헤드 옵션
    classification_heads = [
        LastTokenClassificationHead(),
        FlattenClassificationHead(),
        GlobalAveragePoolingClassificationHead(),
        CLSTokenClassificationHead(),
        FeaturewiseEmbedding(project=False),
        FeaturewiseEmbedding(project=True),
    ]

    # 다양한 트랜스포머 옵션
    transformers = [
        BasicTransformer(2, 128, n_heads=2),
        BasicTransformer(2, 128, n_heads=2, is_decoder=True),
        GPTSmallTransformer(),
        BERTSmallTransformer()
    ]

    # 전처리 설정
    pre_processing = StandardPreProcessing(n_categorical_levels=32)

    # FlowTransformer 정의
    ft = FlowTransformer(
        pre_processing=pre_processing,
        input_encoding=encodings[0],  # NoInputEncoder 사용
        sequential_model=transformers[0],  # BasicTransformer 사용
        classification_head=classification_heads[0],  # LastTokenClassificationHead 사용
        params=FlowTransformerParameters(
            window_size=8, 
            mlp_layer_sizes=[128], 
            mlp_dropout=0.1
        )
    )

    # CICFlowMeter 데이터셋 로드
    dataset_name = "CICFlowMeter_Dataset"
    dataset_specification = NamedDatasetSpecifications.cse_cic_ids_2018_improved
    eval_percent = 0.2  # 20%를 평가용으로 사용
    eval_method = EvaluationDatasetSampling.RandomRows

    print("데이터셋 로딩 중...")
    if cache_folder:
        ft.load_dataset(
            dataset_name, 
            dataset_path, 
            dataset_specification, 
            cache_folder,
            evaluation_dataset_sampling=eval_method, 
            evaluation_percent=eval_percent
        )
    else:
        ft.load_dataset(
            dataset_name, 
            dataset_path, 
            dataset_specification, 
            evaluation_dataset_sampling=eval_method, 
            evaluation_percent=eval_percent
        )

    print("모델 빌드 중...")
    # 모델 빌드
    model = ft.build_model()
    model.summary()

    # 모델 컴파일
    print("모델 컴파일 중...")
    model.compile(
        optimizer="adam", 
        loss='binary_crossentropy', 
        metrics=['binary_accuracy'], 
        jit_compile=True
    )

    # 학습 및 평가
    print("모델 학습 시작...")
    train_results, eval_results, final_epoch = ft.evaluate(
        model, 
        batch_size=128, 
        epochs=10,  # 에포크 수 조정 가능
        steps_per_epoch=64, 
        early_stopping_patience=5
    )

    print("\n=== 학습 완료 ===")
    print("최종 평가 결과:")
    print(eval_results)
    
    return ft, model, train_results, eval_results

def main():
    """
    메인 함수 - 사용자 설정 부분
    """
    
    # 여기에 CICFlowMeter로 생성한 CSV 파일 경로를 입력하세요
    dataset_path = r"C:\path\to\your\cicflowmeter_dataset.csv"
    
    # 캐시 폴더 (선택사항 - 처리 속도 향상을 위해 권장)
    cache_folder = r"C:\path\to\cache\folder"
    
    # 파일 존재 확인
    if not os.path.exists(dataset_path):
        print(f"오류: 데이터셋 파일을 찾을 수 없습니다: {dataset_path}")
        print("dataset_path 변수를 올바른 경로로 설정해주세요.")
        return
    
    # FlowTransformer 실행
    try:
        ft, model, train_results, eval_results = run_cicflowmeter_dataset(
            dataset_path, 
            cache_folder if os.path.exists(cache_folder) else None
        )
        
        print("\n=== 실행 완료 ===")
        print("FlowTransformer 객체와 학습된 모델이 준비되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        print("데이터셋 형식이나 경로를 확인해주세요.")

if __name__ == "__main__":
    main() 