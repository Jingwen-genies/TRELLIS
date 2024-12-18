INPUT_SCHEMA = {
    "image_url": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://github.com/microsoft/TRELLIS/raw/main/assets/example_image/T.png"]
    },
    "task_id": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["1234567890"]
    },
    "return_render": {
        'datatype': 'BOOL',
        'required': True,
        'shape': [1],
        'example': [False]
    },
    "seed": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [0]
    },
    "ss_guidance_strength": {
        'datatype': 'FP64',
        'required': False,
        'shape': [1],
        'example': [7.5]
    },
    "ss_sampling_steps": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [12]
    },
    "slat_guidance_strength": {
        'datatype': 'FP64',
        'required': False,
        'shape': [1],
        'example': [3]
    },
    "slat_sampling_steps": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [12]
    },
    "glb_extraction_simplify": {
        'datatype': 'FP64',
        'required': False,
        'shape': [1],
        'example': [0.0]
    },
    "glb_extraction_texture_size": {
        'datatype': 'INT64',
        'required': False,
        'shape': [1],
        'example': [1024]
    },
    "preprocess_image": {
        'datatype': 'BOOL',
        'required': False,
        'shape': [1],
        'example': [False]
    }
}
